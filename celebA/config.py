
PROJECT_ID = 'example-dev-project-nmrksf0o'
BUCKET_NAME = 'example-datasets-47ml982d'
LABELS = ['Male', 'Young'] # ['Eyeglasses', 'Smiling', 'Attractive', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Necktie']

LABEL = 'Male'

IMAGE_SIZE = 128
ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
       'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
       'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
       'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
       'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
       'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
       'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young']

train_size = 15000
val_size = 5000
test_size = 5000

image_path = 'celebA/img_align_celeba/img_align_celeba'
att_path = 'celebA/list_attr_celeba.csv'
partition_path = 'celebA/list_eval_partition.csv'
landmarks_path = 'celebA/list_landmarks_celeba.csv'
align_landmarks_path = 'celebA/list_landmarks_align_celeba.csv'
celeb_id_path = 'celebA/identity_CelebA.txt'

save_path = "../models/weights"

celeba_face_size = 178