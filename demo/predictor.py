# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import numpy as np
from torch.nn import functional as F

from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
try:
    from visualizer import CustomVisualizer
except:
    from .visualizer import CustomVisualizer


def get_clip_embeddings(vocabulary, prompt='a '):
    from vlpart.modeling.text_encoder.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x.lower().replace(':', ' ') for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

BUILDIN_CLASSIFIER = {
    'pascal_part': 'datasets/metadata/pascal_part_clip_RN50_a+cname.npy',
    'partimagenet': 'datasets/metadata/partimagenet_clip_RN50_a+cname.npy',
    'paco': 'datasets/metadata/paco_clip_RN50_a+cname.npy',
    'lvis': 'datasets/metadata/lvis_v1_clip_RN50_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_RN50_a+cname.npy',
    'voc': 'datasets/metadata/voc_clip_RN50_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'pascal_part': 'pascal_part_val',
    'partimagenet': 'partimagenet_val',
    'paco': 'paco_lvis_v1_val',
    'lvis': 'lvis_v1_val',
    'coco': 'coco_2017_val',
    'voc': 'voc_2007_val',
}

def reset_cls_test(model, cls_path):
    # model.roi_heads.num_classes = num_classes
    if type(cls_path) == str:
        print('Resetting zs_weight', cls_path)
        if cls_path.endswith('npy'):
            zs_weight = np.load(cls_path)
            zs_weight = torch.tensor(
                zs_weight, dtype=torch.float32).permute(1, 0).contiguous()  # dim x C
        elif cls_path.endswith('pth'):
            zs_weight = torch.load(cls_path, map_location='cpu')
            zs_weight = zs_weight.clone().detach().permute(1, 0).contiguous()  # dim x C
        else:
            raise NotImplementedError
        # zs_weight = torch.tensor(
        #     np.load(cls_path),
        #     dtype=torch.float32).permute(1, 0).contiguous() # D x C
    else:
        zs_weight = cls_path
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))],
        dim=1) # D x (C + 1)
    # if model.roi_heads.box_predictor[0].cls_score.norm_weight:
    #     zs_weight = F.normalize(zs_weight, p=2, dim=0)
    # zs_weight = zs_weight.to(model.device)
    # for k in range(len(model.roi_heads.box_predictor)):
    #     del model.roi_heads.box_predictor[k].cls_score.zs_weight
    #     model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight
    zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to(model.device)

    if isinstance(model.roi_heads.box_predictor, torch.nn.ModuleList):
        for idx in range(len(model.roi_heads.box_predictor)):
            model.roi_heads.box_predictor[idx].cls_score.zs_weight_inference = zs_weight
    else:
        model.roi_heads.box_predictor.cls_score.zs_weight_inference = zs_weight


class VisualizationDemo(object):
    def __init__(self, cfg, args=None, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        # self.metadata = MetadataCatalog.get(
        #     cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        # )
        if args is None:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH['pascal_part'])
            classifier = BUILDIN_CLASSIFIER['pascal_part']
        elif args.vocabulary == 'custom':
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        elif args.vocabulary == 'pascal_part_voc':
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        elif args.vocabulary == 'lvis_paco':
            self.metadata = MetadataCatalog.get("__unused")
            lvis_thing_classes = MetadataCatalog.get(
                BUILDIN_METADATA_PATH['lvis']).thing_classes
            paco_thing_classes = MetadataCatalog.get(
                BUILDIN_METADATA_PATH['paco']).thing_classes[75:]
            self.metadata.thing_classes = lvis_thing_classes + paco_thing_classes
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        else:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg
        reset_cls_test(self.predictor.model, classifier)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = CustomVisualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances, args=self.cfg)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
