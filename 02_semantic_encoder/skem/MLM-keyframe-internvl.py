
import logging
import argparse
import pandas as pd
from tqdm import tqdm
import os
import shutil
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from internvl_chat.internvl.conversation import get_conv_template
import types #用于动态绑定方法

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)



def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def custom_chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
                num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
                verbose=False):
    
    if history is None and pixel_values is not None and '<image>' not in question:
        # 如果历史记录为空，并且有图像输入，但问题中没有<image>标记，则在问题前加上<image>标记
        # 问题1 如果一次输入两张图片怎么办？ 答，好像不能自动解决，所以要注意标记数量
        question = '<image>\n' + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list) # 保证输入的图片数量和num_patches_list中的数量一致

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    self.img_context_token_id = img_context_token_id

    template = get_conv_template(self.template)
    template.system_message = self.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

    history = [] if history is None else history
    for (old_question, old_answer) in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        logging.info(f'Image batch size: {image_bs}')

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors='pt')
    #将input_ids解码并且输出出来
    # print("\n\n--log--inputs\n\n",tokenizer.decode(model_inputs['input_ids'][0]))

    input_ids = model_inputs['input_ids'].cuda()
    attention_mask = model_inputs['attention_mask'].cuda()
    #展示完整长度的attention_mask
    # torch.set_printoptions(threshold=10000)  # 设置阈值为 1000，可以根据需要调整
    # print("\n\n--log--attention_mask\n\n",attention_mask)
    generation_config['eos_token_id'] = eos_token_id
    generation_output = self.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config
    )
    scores = generation_output.scores
    sequence_output = generation_output.sequences
    response = tokenizer.batch_decode(sequence_output, skip_special_tokens=True)[0]
    response = response.split(template.sep)[0].strip()
    history.append((question, response))

    # 3. 返回新增 scores 参数
    if return_history:
        return response, history, scores
    else:
        query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
        query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
        if verbose:
            logging.info(query_to_print, response)
        return response, scores

def main(args): 
    model_path = args.model_path
    model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # load_in_4bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()

    model.chat = types.MethodType(custom_chat, model)
    model.cuda()
    logging.info(f"Model loaded from {model_path}")
    logging.info(f"Model is in {model.device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    logging.info(f"Tokenizer loaded from {model_path}")

    generation_config = dict(max_new_tokens=1024, do_sample=False, output_scores=True, return_dict_in_generate=True)

    method = args.method
    threshold = args.threshold

    csv_path = args.csv_path
    df = pd.read_csv(csv_path)
    frame_paths = []
    for index, row in df.iterrows():
        frame_path = row['frame_save_dir']
        frame_paths.append(frame_path)
    
    # 遍历 frame_paths文件夹 列表，对每个文件夹下的.png文件进行操作
    # 1. 读取frames.csv文件，获取所有帧的路径
    # 2. 遍历所有帧的路径，对每一帧进行处理，与模型进行两轮对话，第一轮对话获得对当前帧的描述，第二轮对话获得与对比描述的差异的评价
    # 3. 如果两帧差异大，将当前帧的编号加入到关键帧列表中
    # 4. 将新的关键帧的描述加入到当前对比描述中，继续下一轮对话

    # 开始遍历所有的frame_paths
    for frame_path in frame_paths:
        all_frames = []
        frame_numbers = []
        cur_describe = ""
        cur_frame = ""
        open_frames = pd.read_csv(frame_path + '/frames.csv')
        for index, row in open_frames.iterrows():
            # 获取所有帧的路径
            frame = row['frame_path']
            all_frames.append(frame)
        qbar = tqdm(total = len(all_frames), desc = 'Processing frames')

        for frame_file in all_frames:
            if cur_frame == "": # 第一帧
                cur_frame = frame_file
                frame_number = frame_file.split('/')[-1].split('.')[0]
                frame_numbers.append(frame_number)
                #将关键帧保存到关键帧文件夹中
                key_frame_path = frame_path + '/key_frames' + method
                os.makedirs(key_frame_path, exist_ok=True)
                key_frame_file = key_frame_path + '/' + frame_number + '.png'
                shutil.copy(frame_file, key_frame_file)
                logging.info(f"frame_file: {frame_file.split('/')[-2]}")
                continue
            logging.info(f"frame_index: {frame_file.split('/')[-1].split('.')[0]}")

            # 读取新一帧的图片
            pixel_values1 = load_image(cur_frame, max_num=12).to(torch.bfloat16).cuda()
            pixel_values2 = load_image(frame_file, max_num=12).to(torch.bfloat16).cuda()
            pixel_values = torch.cat([pixel_values1, pixel_values2], dim=0)
            num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

            question1 = args.q1
            question2 = args.q2

            # 第一轮对话
            response, history, _ = model.chat(
                tokenizer,
                pixel_values,
                question1,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
            logging.info(f"response1: {response}")

            # 第二轮对话
            response2, history, scores = model.chat(
                tokenizer,
                pixel_values,
                question2,
                generation_config,
                num_patches_list=num_patches_list,
                history=history,
                return_history=True,
            )
            logging.info(f"response2: {response2}")

            # 处理第二轮对话的结果
            no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
            no_prob = torch.softmax(scores[0], dim=-1)[0, no_token_id].item()
            logging.info(f"no_prob: {no_prob}")
            yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
            yes_prob = torch.softmax(scores[0], dim=-1)[0, yes_token_id].item()
            logging.info(f"yes_prob: {yes_prob}")

            # 计算no_prob和yes_prob的差值
            diff = no_prob - yes_prob
            if diff > threshold:
                logging.warning(f"diff: {diff}")
            else:
                logging.info(f"diff: {diff}")

            # 计算no_prob和yes_prob的比值
            ratio = no_prob / yes_prob
            logging.info(f"ratio: {ratio}")

            # if no_prob >0.5:
            if diff > threshold:
                cur_frame = frame_file
                cur_describe = response
                frame_number = frame_file.split('/')[-1].split('.')[0]
                frame_numbers.append(frame_number)
                #将关键帧保存到关键帧文件夹中
                key_frame_path = frame_path + '/key_frames' + method
                os.makedirs(key_frame_path, exist_ok=True)
                key_frame_file = key_frame_path + '/' + frame_number + '.png'
                shutil.copy(frame_file, key_frame_file)
            logging.info(f"frame_numbers: {frame_numbers}")
            qbar.update(1)

        #将最后一帧也当作关键帧拷贝到关键帧文件夹中
        frame_file = all_frames[-1]
        frame_number = frame_file.split('/')[-1].split('.')[0]
        key_frame_path = frame_path + '/key_frames' + method
        os.makedirs(key_frame_path, exist_ok=True)
        key_frame_file = key_frame_path + '/' + frame_number + '.png'
        logging.info(f"coping {frame_file} to {key_frame_file}")
        shutil.copy(frame_file, key_frame_file)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 两次对话的prompt
    #Highlight notable similarities or differences while 
    prompt_ask_image = '''Image-1: <image>\nImage-2: <image>\n**Compare the two images.** Provide a detailed description for each image, focusing on the overall setting, key objects, their positions, gestures, colors, and any movements. Keeping the descriptions distinct. Use clear, everyday language and follow this format: `img1{ [description of first image] } img2{ [description of second image] }`.'''
    prompt_compare_image='''**Compare the two descriptions of the images you have given.** Focus on the semantic similarity of the images: the positions of key objects in the scene, any objects that have appeared or disappeared and the extent of changes in the background environment. Determine if these aspects depict the exact same scene. **Only respond with "yes" if they match, otherwise respond with "no".**'''
    parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL2-8B')
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--method", type=str, default="intervl")
    parser.add_argument("--q1", type=str, default=prompt_ask_image)
    parser.add_argument("--q2", type=str, default=prompt_compare_image)
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="PSSS semantic-divergence threshold eta_th for keyframe selection (default 0.35)")
    args = parser.parse_args()
    # 获取当前时间
    import datetime
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s | %(message)s',
    handlers=[
    logging.FileHandler(f"{args.method}_{time}.log"),
    logging.StreamHandler()
])
    main(args)
