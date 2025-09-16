from utils.trainers.models import ViT_Multitask
import os
import yaml
import json
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from utils.data_preprocessing.func_dataset_multitask import ResizeMin
import matplotlib.pyplot as plt
import predict_singletask


# Predict Single Image
def predict(image_path, config, label_ent=0, label_disease=0):

    processor = transforms.Compose([
        ResizeMin(226),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    MODEL = ViT_Multitask(config, config)
    MODEL.load_state_dict(torch.load(config['model_multitask'], map_location=config['map_location']))

    img = Image.open(image_path).convert('RGB')
    pro_img = processor(img)
    batch_img = pro_img.unsqueeze(0)

    MODEL.eval()
    out = MODEL(batch_img)

    prob1 = F.softmax(out[0], dim=1)
    prob2 = F.softmax(out[1], dim=1)

    max_index1 = torch.argmax(prob1, dim=1)
    max_index2 = torch.argmax(prob2, dim=1)

    if( max_index1[0] == label_ent ):
        bol1 = True
    else:
        bol1 = False
    if( max_index2[0] == label_disease ):
        bol2 = True
    else:
        bol2 = False

    return config['list tasks'][max_index1[0]], config['list diseases'][max_index2[0]], bol1, bol2

# Predict Top 2 Single Image
def predict_topk(image_path, config, label_ent=0, label_disease=0):

    processor = transforms.Compose([
        ResizeMin(226),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    MODEL = ViT_Multitask(config, config)
    MODEL.load_state_dict(torch.load(config['model_multitask'], map_location=config['map_location']))

    img = Image.open(image_path).convert('RGB')
    pro_img = processor(img)
    batch_img = pro_img.unsqueeze(0)

    MODEL.eval()
    out = MODEL(batch_img)

    prob1 = F.softmax(out[0], dim=1)
    prob2 = F.softmax(out[1], dim=1)

    max_index1 = torch.argmax(prob1, dim=1)
    top2 = torch.topk(prob2, k=2).indices

    if( max_index1[0] == label_ent ):
        bol1 = True
    else:
        bol1 = False
    if( (top2[0][0] == label_disease) or (top2[0][1] == label_disease) ):
        bol2 = True
    else:
        bol2 = False

    return config['list tasks'][max_index1[0]], config ['list diseases'][top2[0][0]], config ['list diseases'][top2[0][1]], bol1, bol2

# Predict Folder Image
def predict_fol_images(config, path_fol_images, label_ent, label_disease):
    lists = os.listdir(path_fol_images)
    all = []
    for i in lists:
        path_img = os.path.join(path_fol_images, i)
        a, b, bol1, bol2 = predict(path_img, config, label_ent, label_disease)
        if ((bol1 == True) and (bol2 == True)):
            all.append(1)
            print('CORRECT')
            print('\t{}'.format(i))
            print('\t{}: {}\n'.format(a, b))
        else:
            all.append(0)
            print('ERROR')
            print('\t{}'.format(i))
            print('\t{}: {}\n'.format(a, b))
    print('Number images correct:', sum(all))
    print('Total images:', len(all))
    print('Acuracy: {:.2f}%'.format(sum(all)/len(all)*100.0))
    return "{:.2f}".format(sum(all)/len(all)*100.0)

# Predict Polder Image with top 2
def predict_fol_images_topk(config, path_fol_images, label_ent, label_disease):
    lists = os.listdir(path_fol_images)
    all = []
    for i in lists:
        path_img = os.path.join(path_fol_images, i)
        a, b1, b2, bol1, bol2 = predict_topk(path_img, config, label_ent, label_disease)
        if ((bol1 and bol2) == True):
            all.append(1)
            print('CORRECT')
            print('\t{}'.format(i))
            print('\t{}: 1.{}, 2.{}\n'.format(a, b1, b2))
        else:
            all.append(0)
            print('ERROR')
            print('\t{}'.format(i))
            print('\t{}: {}\n'.format(a, b1))
    print('Number images correct:', sum(all))
    print('Total images:', len(all))
    print('Acuracy: {:.2f}%'.format(sum(all)/len(all)*100.0))
    return "{:.2f}".format(sum(all)/len(all)*100.0)

# Save image predict False
def save_image_predict(config, path_fol_images, output_save):
    list_tasks = config['list tasks']
    list_diseases = config['list diseases']

    ent = os.listdir(path_fol_images)
    for i1 in ent:
        print(i1)
        # 1
        p1 = os.path.join(output_save, i1)
        if not os.path.exists(p1):
            os.makedirs(p1)
        # 2
        c1 = os.path.join(path_fol_images, i1)
        diseases = os.listdir(c1)
        for i2 in diseases:
            print(i2)
            # 1
            p2 = os.path.join(p1, i2)
            if not os.path.exists(p2):
                os.makedirs(p2)
            # 2
            c2 = os.path.join(c1, i2)
            images = os.listdir(c2)
            for i3 in images:
                path_img = os.path.join(c2, i3)
                a, b, bol1, bol2 = predict(path_img, config, list_tasks.index(str(i1)), list_diseases.index(str(i2)))
                if (bol1 == False or bol2 == False):
                    print('ERROR')
                    print('\t{}'.format(i3))
                    print('\t{}: {}\n'.format(a, b))
                    # Plot
                    text1 = "Target: {}".format(str(i2))
                    text2 = "Predict: {}".format(b)
                    image = Image.open(path_img)
                    fig, ax = plt.subplots()
                    ax.imshow(image)
                    ax.axis('off')
                    ax.text(0.5, 0.95, text1, fontsize=15, ha='center', va='center', transform=ax.transAxes, color='green')
                    ax.text(0.5, 0.86, text2, fontsize=15, ha='center', va='center', transform=ax.transAxes, color='blue')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    output_path = os.path.join(p2, i3)
                    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print('DONE')

# Combine 2 two model multitask and single
def combine_2_model(config_mul, config_sig, path_fol_images, label_ent, label_disease):
    list_diseases = config_mul['list diseases']

    lists = os.listdir(path_fol_images)
    all = []
    for i in lists:
        path_img = os.path.join(path_fol_images, i)
        a, b, bol1, bol2 = predict(path_img, config_mul, label_ent, label_disease)
        out = predict_singletask.predict_print(path_img, config_sig)

        if( (b == list_diseases[label_disease]) or (out == list_diseases[label_disease]) ):
            bol2 = True
        else:
            bol2 = False

        if ((bol1 == True) and (bol2 == True)):
            all.append(1)
            print('CORRECT')
            print('\t{}'.format(i))
            print('\t{}: {} - {}\n'.format(a, b, out))
        else:
            all.append(0)
            print('ERROR')
            print('\t{}'.format(i))
            print('\t{}: {} - {}\n'.format(a, b, out))
    print('Number images correct:', sum(all))
    print('Total images:', len(all))
    print('Acuracy: {:.2f}%'.format(sum(all)/len(all)*100.0))
    return "{:.2f}".format(sum(all)/len(all)*100.0)

# Load config
with open('/home/bht/pycharm/PCT/config/config_predict_multitask.yaml') as file:
    config_mul = yaml.safe_load(file)
with open('/home/bht/pycharm/PCT/config/predict_singletask.yaml') as file:
    config_sig = yaml.safe_load(file)

path_fol = "/home/bht/VKU/ThucTap_AI_Y_Te_PCT/data/Ear_Nose_Throat/val"
ent = os.listdir(path_fol)
list_tasks = config_mul['list tasks']
list_diseases = config_mul['list diseases']
lists = []
for i1 in ent:
    print(i1)
    p1 = os.path.join(path_fol, i1)
    l1 = os.listdir(p1)
    for i2 in l1:
        print('\t', i2)
        p2 = os.path.join(p1, i2)
        acc = combine_2_model(config_mul, config_sig, p2, list_tasks.index(str(i1)), list_diseases.index(str(i2)))
        text = "Accuracy of " + str(i2) + " = " + str(acc) + "%"
        lists.append(text)

output_save = '/home/bht/Downloads/accuracy.json'
with open(output_save, "w") as file:
    json.dump(lists, file)
