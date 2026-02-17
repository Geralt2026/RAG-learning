import time
from typing import Dict, Any, Optional


class CacheManager:
    """简单的缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存生存时间(秒)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _generate_key(self, query: str, thread_id: Optional[str] = None) -> str:
        """
        生成缓存键
        
        Args:
            query: 查询内容
            thread_id: 线程ID(可选)
            
        Returns:
            str: 缓存键
        """
        if thread_id:
            return f"{thread_id}:{query}"
        return query
    
    def get(self, query: str, thread_id: Optional[str] = None) -> Optional[Any]:
        """
        获取缓存内容
        
        Args:
            query: 查询内容
            thread_id: 线程ID(可选)
            
        Returns:
            Any: 缓存内容，未命中则返回None
        """
        key = self._generate_key(query, thread_id)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # 检查缓存是否过期
            if time.time() - entry["timestamp"] <= self.ttl_seconds:
                # 更新访问时间
                entry["last_access"] = time.time()
                return entry["value"]
            
            # 过期，删除缓存
            del self.cache[key]
        
        return None
    
    def set(self, query: str, value: Any, thread_id: Optional[str] = None, quality: float = 0.5) -> None:
        """
        设置缓存内容
        
        Args:
            query: 查询内容
            value: 缓存值
            thread_id: 线程ID(可选)
            quality: 内容质量评分(0.0-1.0)
        """
        key = self._generate_key(query, thread_id)
        
        # 检查缓存是否已满，如果已满则清理最旧的或评分最低的条目
        if len(self.cache) >= self.max_size:
            self._evict_cache()
        
        # 添加新缓存
        self.cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "last_access": time.time(),
            "quality": quality
        }
    
    def delete(self, query: str, thread_id: Optional[str] = None) -> bool:
        """
        删除缓存内容
        
        Args:
            query: 查询内容
            thread_id: 线程ID(可选)
            
        Returns:
            bool: 是否成功删除
        """
        key = self._generate_key(query, thread_id)
        
        if key in self.cache:
            del self.cache[key]
            return True
        
        return False
    
    def update_quality(self, query: str, quality: float, thread_id: Optional[str] = None) -> bool:
        """
        更新缓存内容质量评分
        
        Args:
            query: 查询内容
            quality: 新的质量评分(0.0-1.0)
            thread_id: 线程ID(可选)
            
        Returns:
            bool: 是否成功更新
        """
        key = self._generate_key(query, thread_id)
        
        if key in self.cache:
            self.cache[key]["quality"] = max(0.0, min(1.0, quality))
            return True
        
        return False
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
    
    def _evict_cache(self) -> None:
        """清理缓存，移除最旧的或质量最低的条目"""
        if not self.cache:
            return
        
        # 根据最后访问时间和质量评分计算优先级
        entries = [(k, v["last_access"], v["quality"]) for k, v in self.cache.items()]
        
        # 优先移除质量低的，如果质量相近，则移除最旧的
        entries.sort(key=lambda x: (x[2], x[1]))
        
        # 移除优先级最低的条目
        if entries:
            del self.cache[entries[0][0]]


# 创建全局实例
cache_manager = CacheManager()