# Copyright (c) Facebook, Inc. and its affiliates.

# imagenet_id starts from 1
PASCAL_GOLDEN_CATEGORIES = [
  # pascal_part
    {'id': 1, 'name': 'aeroplane', 'imagenet_id': [[405, 405]]},
    {'id': 2, 'name': 'bicycle', 'imagenet_id': [[445, 445], [672, 672]]},
    {'id': 3, 'name': 'bird', 'imagenet_id': [[8, 25], [81, 101]]},
    {'id': 4, 'name': 'boat', 'imagenet_id': [[485, 485]]},
    {'id': 5, 'name': 'bottle', 'imagenet_id': [[441, 441]]},
    {'id': 6, 'name': 'bus', 'imagenet_id': [[780, 780]]},
    {'id': 7, 'name': 'car', 'imagenet_id': [[408, 408], [610, 610], [628, 628], [818, 818]]},
    {'id': 8, 'name': 'cat', 'imagenet_id': [[282, 286]]},
    {'id': 9, 'name': 'chair', 'imagenet_id': [[424, 424]]},
    {'id': 10, 'name': 'cow', 'imagenet_id': [[346, 348]]},
    {'id': 11, 'name': 'dining table', 'imagenet_id': [[533, 533]]},
    {'id': 12, 'name': 'dog', 'imagenet_id': [[152, 277]]},
    {'id': 13, 'name': 'horse', 'imagenet_id': []},
    {'id': 14, 'name': 'motorbike', 'imagenet_id': [[671, 671]]},
    {'id': 15, 'name': 'person', 'imagenet_id': []},
    {'id': 16, 'name': 'potted plant', 'imagenet_id': []},
    {'id': 17, 'name': 'sheep', 'imagenet_id': [[349, 354]]},
    {'id': 18, 'name': 'sofa', 'imagenet_id': []},
    {'id': 19, 'name': 'train', 'imagenet_id': [[467, 467]]},
    {'id': 20, 'name': 'tv monitor', 'imagenet_id': [[762, 762]]},
]

PARTIMAGENET_GOLDEN_CATEGORIES = [
    # partimagenet
    {"id": 101, "name": "Quadruped", 'imagenet_id': [[102, 107], [270, 298], [341, 357]]}, 
    {"id": 102, "name": "Biped", 'imagenet_id': [[366, 385]]}, 
    {"id": 103, "name": "Fish", 'imagenet_id': [[1, 7], [390, 398]]},
    {"id": 104, "name": "Bird", 'imagenet_id': [[81, 101]]},
    {'id': 105, 'name': 'Snake', 'imagenet_id': [[53, 69]]},
    {"id": 106, "name": "Reptile", 'imagenet_id': [[26, 52]]},
    {"id": 107, "name": "Car", 'imagenet_id': [[408, 408], [610, 610], [628, 628], [818, 818]]}, 
    {"id": 108, "name": "Bicycle", 'imagenet_id': [[445, 445], [672, 672]]}, 
    {"id": 109, "name": "Boat", 'imagenet_id': [[485, 485]]}, 
    {"id": 110, "name": "Aeroplane", 'imagenet_id': [[405, 405]]}, 
    {"id": 111, "name": "Bottle", 'imagenet_id': [[441, 441]]}, 
]


PACO_GOLDEN_CATEGORIES = [
    # paco
    {'id': 300, 'name': 'trash_can', 'imagenet_id': []},
    {'id': 301, 'name': 'handbag', 'imagenet_id': []},
    {'id': 302, 'name': 'ball', 'imagenet_id': []},
    {'id': 303, 'name': 'basket', 'imagenet_id': []},
    {'id': 304, 'name': 'belt', 'imagenet_id': []},
    {'id': 305, 'name': 'bench', 'imagenet_id': [[704, 704]]},
    {'id': 306, 'name': 'bicycle', 'imagenet_id': [[445, 445], [672, 672]]}, 
    {'id': 307, 'name': 'blender', 'imagenet_id': []},
    {'id': 308, 'name': 'book', 'imagenet_id': []},
    {'id': 309, 'name': 'bottle', 'imagenet_id': [[441, 441]]},
    {'id': 310, 'name': 'bowl', 'imagenet_id': [[810, 810]]},
    {'id': 311, 'name': 'box', 'imagenet_id': []},
    {'id': 312, 'name': 'broom', 'imagenet_id': [[463, 463]]},
    {'id': 313, 'name': 'bucket', 'imagenet_id': [[464, 464]]},
    {'id': 314, 'name': 'calculator', 'imagenet_id': []},
    {'id': 315, 'name': 'can', 'imagenet_id': []},
    {'id': 316, 'name': 'car_(automobile)', 'imagenet_id': [[408, 408], [610, 610], [628, 628], [818, 818]]},
    {'id': 317, 'name': 'carton', 'imagenet_id': [[479, 479]]},
    {'id': 318, 'name': 'cellular_telephone', 'imagenet_id': [[488, 488]]},
    {'id': 319, 'name': 'chair', 'imagenet_id': [[424, 424]]},
    {'id': 320, 'name': 'clock', 'imagenet_id': [[893, 893]]},
    {'id': 321, 'name': 'crate', 'imagenet_id': [[520, 520]]},
    {'id': 322, 'name': 'cup', 'imagenet_id': [[969, 969]]},
    {'id': 323, 'name': 'dog', 'imagenet_id': [[152, 160]]}, # [[152, 277]] too many
    {'id': 324, 'name': 'drill', 'imagenet_id': [[741, 741]]},
    {'id': 325, 'name': 'drum_(musical_instrument)', 'imagenet_id': []},
    {'id': 326, 'name': 'earphone', 'imagenet_id': []},
    {'id': 327, 'name': 'fan', 'imagenet_id': [[546, 546]]},
    {'id': 328, 'name': 'glass_(drink_container)', 'imagenet_id': []},
    {'id': 329, 'name': 'guitar', 'imagenet_id': [[547, 547]]},
    {'id': 330, 'name': 'hammer', 'imagenet_id': [[588, 588]]},
    {'id': 331, 'name': 'hat', 'imagenet_id': []},
    {'id': 332, 'name': 'helmet', 'imagenet_id': [[519, 519]]},
    {'id': 333, 'name': 'jar', 'imagenet_id': []},
    {'id': 334, 'name': 'kettle', 'imagenet_id': []},
    {'id': 335, 'name': 'knife', 'imagenet_id': []},
    {'id': 336, 'name': 'ladder', 'imagenet_id': []},
    {'id': 337, 'name': 'lamp', 'imagenet_id': [[847, 847]]},
    {'id': 338, 'name': 'laptop_computer', 'imagenet_id': [[621, 621]]},
    {'id': 339, 'name': 'microwave_oven', 'imagenet_id': [[652, 652]]},
    {'id': 340, 'name': 'mirror', 'imagenet_id': []},
    {'id': 341, 'name': 'mouse_(computer_equipment)', 'imagenet_id': [[674, 674]]},
    {'id': 342, 'name': 'mug', 'imagenet_id': [[505, 505]]},
    {'id': 343, 'name': 'napkin', 'imagenet_id': []},
    {'id': 344, 'name': 'newspaper', 'imagenet_id': []},
    {'id': 345, 'name': 'pan_(for_cooking)', 'imagenet_id': [[568, 568]]},
    {'id': 346, 'name': 'pen', 'imagenet_id': []},
    {'id': 347, 'name': 'pencil', 'imagenet_id': []},
    {'id': 348, 'name': 'pillow', 'imagenet_id': [[722, 722]]},
    {'id': 349, 'name': 'pipe', 'imagenet_id': []},
    {'id': 350, 'name': 'plate', 'imagenet_id': []},
    {'id': 351, 'name': 'pliers', 'imagenet_id': []},
    {'id': 352, 'name': 'remote_control', 'imagenet_id': [[762, 762]]},
    {'id': 353, 'name': 'plastic_bag', 'imagenet_id': [[729, 729]]},
    {'id': 354, 'name': 'scarf', 'imagenet_id': []},
    {'id': 355, 'name': 'scissors', 'imagenet_id': []},
    {'id': 356, 'name': 'screwdriver', 'imagenet_id': [[785, 785]]},
    {'id': 357, 'name': 'shoe', 'imagenet_id': [[771, 771]]},
    {'id': 358, 'name': 'slipper_(footwear)', 'imagenet_id': []},
    {'id': 359, 'name': 'soap', 'imagenet_id': []},
    {'id': 360, 'name': 'sponge', 'imagenet_id': []},
    {'id': 361, 'name': 'spoon', 'imagenet_id': []},
    {'id': 362, 'name': 'stool', 'imagenet_id': []},
    {'id': 363, 'name': 'sweater', 'imagenet_id': []},
    {'id': 364, 'name': 'table', 'imagenet_id': [[533, 533]]},
    {'id': 365, 'name': 'tape_(sticky_cloth_or_paper)', 'imagenet_id': []},
    {'id': 366, 'name': 'telephone', 'imagenet_id': [[529, 529]]},
    {'id': 367, 'name': 'television_set', 'imagenet_id': [[852, 852]]},
    {'id': 368, 'name': 'tissue_paper', 'imagenet_id': []},
    {'id': 369, 'name': 'towel', 'imagenet_id': []},
    {'id': 370, 'name': 'tray', 'imagenet_id': [[869, 869]]},
    {'id': 371, 'name': 'vase', 'imagenet_id': [[884, 884]]},
    {'id': 372, 'name': 'wallet', 'imagenet_id': [[894, 894]]},
    {'id': 373, 'name': 'watch', 'imagenet_id': []},
    {'id': 374, 'name': 'wrench', 'imagenet_id': []},
]

GOLDEN_CATEGORIES = PASCAL_GOLDEN_CATEGORIES + PARTIMAGENET_GOLDEN_CATEGORIES + PACO_GOLDEN_CATEGORIES

ADDITIONAL_CATEGORIES = [
    {"id": 501, "name": "elephant", 'imagenet_id': [[102, 102], [386, 387]]},
    {"id": 502, "name": "koala", 'imagenet_id': [[106, 106]]},
    {"id": 503, "name": "fox", 'imagenet_id': [[278, 281]]},
    {"id": 504, "name": "leopard", 'imagenet_id': [[287, 290], [294, 294]]},
    {"id": 505, "name": "tiger", 'imagenet_id': [[291, 291], [293, 293]]},
    {"id": 506, "name": "lion", 'imagenet_id': [[292, 292]]},
    {"id": 507, "name": "bear", 'imagenet_id': [[295, 298]]},
    {"id": 508, "name": "zebra", 'imagenet_id': [[341, 341]]},
    {"id": 509, "name": "pig", 'imagenet_id': [[342, 344]]},
    {"id": 510, "name": "monkey", 'imagenet_id': [[374, 385]]},
    {"id": 511, "name": "panda", 'imagenet_id': [[388, 389]]},
]


def _get_builtin_metadata():
    golden_categories = GOLDEN_CATEGORIES + ADDITIONAL_CATEGORIES
    id_to_name = {x['id']: x['name'] for x in golden_categories}
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(
            sorted(golden_categories, key=lambda x: x['id']))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}
