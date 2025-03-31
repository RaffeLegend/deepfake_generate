import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextSimilarity:
    def __init__(self, method='cosine'):
        """
        初始化文本相似度类
        :param method: 指定相似度计算方法, 默认使用'cosine'余弦相似度
        """
        self.method = method
        self.vectorizer = TfidfVectorizer()

    # Load data from Json
    def load_data(self, prompt_path, prompt_index):
        info_list = list()
        for root, _, files in os.walk(prompt_path):
            for file in files:
                info_path = os.path.join(root, file)
                info_list.append(info_path)

        info_list.sort()

        for index, path in enumerate(info_list):
            if prompt_index in path:
                self.data_sets = info_list[index:]
                break

        return info_list
    
    def load_json(self, prompt_json):
        with open(prompt_json, 'r') as f:
            data = json.load(f)
        return data
    
    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            output_path = self.get_output_path(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                prompt = self.prompt_process(prompt, NEGATIVE_PROMPT)
                image = self.model(
                            prompt=prompt,
                            num_inference_steps=50,
                            guidance_scale=7.0,
                            ).images[0]
                save_image(image, output_path, index)

    def fit_transform(self, documents):
        """
        将输入文本转化为TF-IDF矩阵
        :param documents: 输入文本列表
        :return: TF-IDF矩阵
        """
        return self.vectorizer.fit_transform(documents)

    def calculate_similarity(self, documents):
        """
        计算输入文本之间的相似度
        :param documents: 输入文本列表
        :return: 文本之间的相似度矩阵
        """
        tfidf_matrix = self.fit_transform(documents)
        if self.method == 'cosine':
            return cosine_similarity(tfidf_matrix)

    def most_similar(self, documents, index, top_n=3):
        """
        获取指定文本与其他文本最相似的top_n文本
        :param documents: 输入文本列表
        :param index: 需要查找相似文本的文本索引
        :param top_n: 返回前n个最相似的文本
        :return: 最相似的文本索引列表
        """
        similarity_matrix = self.calculate_similarity(documents)
        similarity_scores = similarity_matrix[index]
        most_similar_indices = similarity_scores.argsort()[-top_n:][::-1]
        return most_similar_indices[1:]  # 排除自己本身