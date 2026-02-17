import re
import shutup
shutup.please()
import jieba
import jieba.analyse


def extract_smart_keywords(query):
    """
    使用jieba智能提取中文关键词
    
    Args:
        query: 用户查询文本
        
    Returns:
        list: 提取的关键词列表
    """
    if not query:
        return []
        
    try:        
        # 常见的停用词
        stop_words = ['什么', '之间', '发生', '关系', '的', '和', '有', '是', '在', '了', '吗',
                     '为什么', '如何', '怎么', '怎样', '请问', '告诉', '我', '你', '他', '她', '它',
                     '们', '这个', '那个', '这些', '那些', '一个', '一些', '一下', '地', '得', '着']
        
        # 使用TF-IDF提取关键词
        tfidf_keywords = jieba.analyse.extract_tags(query, topK=3)
        
        # 使用TextRank提取关键词
        textrank_keywords = jieba.analyse.textrank(query, topK=3)
        
        # 使用精确模式分词提取2个字以上的词
        seg_list = jieba.cut(query, cut_all=False)
        seg_words = [word for word in seg_list if len(word) >= 2 and word not in stop_words]
        
        # 合并关键词并去重
        all_keywords = list(set(tfidf_keywords + textrank_keywords + seg_words))
        
        # 按长度排序，优先使用长词
        all_keywords.sort(key=len, reverse=True)
        
        # 如果关键词超过5个，只取前5个
        result = all_keywords[:5] if len(all_keywords) > 5 else all_keywords
        
        # 如果没有提取到关键词，尝试直接提取实体名称
        if not result:
            # 匹配实体名称
            entity_names = re.findall(r'[\u4e00-\u9fa5]{2,}', query)
            result = [name for name in entity_names if name not in stop_words]
        
        return result
        
    except ImportError:
        print("jieba库未安装，使用简单分词")
        # 回退到简单的正则匹配
        words = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]{2,}', query)
        stop_words = ['什么', '之间', '发生', '关系', '的', '和', '有', '是', '在', '了', '吗']
        return [w for w in words if w not in stop_words]
    except Exception as e:
        print(f"关键词提取失败: {e}")
        # 最后回退到直接分割
        return [query]