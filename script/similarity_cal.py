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

# 使用示例
documents = [
    "This is a sample document.",
    "This document is another example.",
    "We are testing the text similarity algorithm.",
    "This is another sample document for testing."
]

similarity_checker = TextSimilarity()
similarity_matrix = similarity_checker.calculate_similarity(documents)

# 输出相似度矩阵
print("Similarity Matrix:")
print(similarity_matrix)

# 获取文档 0 的最相似文本
most_similar_docs = similarity_checker.most_similar(documents, index=0, top_n=3)
print(f"Most similar documents to document 0: {most_similar_docs}")
