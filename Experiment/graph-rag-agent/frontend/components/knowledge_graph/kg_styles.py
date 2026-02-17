# 知识图谱CSS样式
KG_STYLES = """
<style>
    .vis-network {
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        position: relative;
    }
    .vis-tooltip {
        background-color: white !important;
        color: #333 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 4px !important;
        padding: 8px !important;
        font-family: 'Arial', sans-serif !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    /* 增加节点悬停动画效果 */
    .vis-node:hover {
        transform: scale(1.1);
        transition: all 0.3s ease;
    }
    
    /* Neo4j风格右键菜单样式 */
    .node-context-menu {
        position: absolute;
        background: white;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 5px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
        min-width: 150px;
    }
    
    .node-context-menu-item {
        padding: 5px 10px;
        cursor: pointer;
    }
    
    .node-context-menu-item:hover {
        background-color: #f0f0f0;
    }
    
    .node-context-menu-header {
        padding: 5px 10px;
        font-weight: bold;
        border-bottom: 1px solid #eee;
    }
    
    /* 控制面板样式 */
    .graph-control-panel {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 999;
        background: white;
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        min-width: 180px;
    }
    
    .graph-control-button {
        display: block;
        width: 100%;
        margin: 5px 0;
        padding: 6px 12px;
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 4px;
        cursor: pointer;
        text-align: left;
    }
    
    .graph-control-button:hover {
        background-color: #e9ecef;
    }
    
    .graph-info {
        font-size: 12px;
        margin-top: 8px;
        color: #666;
        border-top: 1px solid #eee;
        padding-top: 8px;
    }
</style>
"""