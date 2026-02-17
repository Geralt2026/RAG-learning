import tempfile
import os
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
from frontend_config.settings import KG_COLOR_PALETTE, NODE_TYPE_COLORS

def visualize_knowledge_graph(kg_data):
    """使用pyvis可视化知识图谱 - 动态节点类型和颜色，支持Neo4j式交互"""
    if not kg_data or "nodes" not in kg_data or "links" not in kg_data:
        st.warning("无法获取知识图谱数据")
        return
    
    if len(kg_data["nodes"]) == 0:
        st.info("没有找到相关的实体和关系")
        return
    
    # 添加图表设置控制 - 增加交互说明
    with st.expander("图谱显示设置与交互说明", expanded=False):
        st.markdown("""
        ### 交互说明
        - **双击节点**: 聚焦查看该节点及其直接相连的节点和关系
        - **右键节点**: 打开上下文菜单，提供更多操作
        - **单击空白处**: 重置图谱，显示所有节点
        - **使用控制面板**: 右上角的控制面板提供重置和返回上一步功能
        
        ### 显示设置
        """)
        
        # 为每个checkbox添加唯一的key参数
        # 通过使用随机生成或基于kg_data一部分内容的哈希值创建唯一键
        import hashlib
        
        # 基于kg_data的节点数量和时间戳创建哈希值的一部分
        import time
        timestamp = str(time.time())
        node_count = str(len(kg_data["nodes"]))
        base_key = hashlib.md5((node_count + timestamp).encode()).hexdigest()[:8]
        
        col1, col2 = st.columns(2)
        with col1:
            physics_enabled = st.checkbox("启用物理引擎", 
                                       value=st.session_state.kg_display_settings["physics_enabled"],
                                       key=f"physics_enabled_{base_key}",
                                       help="控制节点是否可以动态移动")
            node_size = st.slider("节点大小", 10, 50, 
                                st.session_state.kg_display_settings["node_size"],
                                key=f"node_size_{base_key}",
                                help="调整节点的大小")
        
        with col2:
            edge_width = st.slider("连接线宽度", 1, 10, 
                                 st.session_state.kg_display_settings["edge_width"],
                                 key=f"edge_width_{base_key}", 
                                 help="调整连接线的宽度")
            spring_length = st.slider("弹簧长度", 50, 300, 
                                    st.session_state.kg_display_settings["spring_length"],
                                    key=f"spring_length_{base_key}", 
                                    help="调整节点之间的距离")
        
        # 更新设置
        st.session_state.kg_display_settings = {
            "physics_enabled": physics_enabled,
            "node_size": node_size,
            "edge_width": edge_width,
            "spring_length": spring_length,
            "gravity": st.session_state.kg_display_settings["gravity"]
        }
    
    # 创建网络图 - 修改背景为白色
    net = Network(height="600px", width="100%", bgcolor="#FFFFFF", font_color="#333333", directed=True)
    
    # 增强配置 - 为Neo4j式交互添加配置
    net.set_options("""
    {
      "physics": {
        "enabled": %s,
        "barnesHut": {
          "gravitationalConstant": %d,
          "centralGravity": 0.5,
          "springLength": %d,
          "springConstant": 0.04,
          "damping": 0.15,
          "avoidOverlap": 0.1
        },
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 100,
          "onlyDynamicEdges": false,
          "fit": true
        }
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": {
          "enabled": true,
          "bindToWindow": true
        },
        "hover": true,
        "multiselect": true,
        "tooltipDelay": 200
      },
      "layout": {
        "improvedLayout": true,
        "hierarchical": {
          "enabled": false
        }
      }
    }
    """ % (str(physics_enabled).lower(), st.session_state.kg_display_settings["gravity"], spring_length))
    
    # 提取所有唯一组类型
    group_types = set()
    for node in kg_data["nodes"]:
        group = node.get("group", "Unknown")
        if group:
            group_types.add(group)
    
    # 为每个组分配颜色
    group_colors = {}
    
    # 首先分配预定义颜色
    for group in group_types:
        if group in NODE_TYPE_COLORS:
            group_colors[group] = NODE_TYPE_COLORS[group]
    
    # 然后为剩余组分配颜色
    palette_index = 0
    for group in sorted(group_types):
        if group not in group_colors:
            # 处理社区类型
            if isinstance(group, str) and "Community" in group:
                try:
                    # 提取社区ID数字部分
                    comm_id_str = group.replace("Community", "")
                    if not comm_id_str:
                        comm_id = 0
                    else:
                        comm_id = int(comm_id_str)
                    
                    # 确保使用一致的社区颜色映射
                    color_index = (comm_id - 1) % len(KG_COLOR_PALETTE) if comm_id > 0 else 0
                    group_colors[group] = KG_COLOR_PALETTE[color_index]
                except (ValueError, TypeError):
                    # 转换失败，使用默认分配
                    group_colors[group] = KG_COLOR_PALETTE[palette_index % len(KG_COLOR_PALETTE)]
                    palette_index += 1
            else:
                # 普通类型按序分配颜色
                group_colors[group] = KG_COLOR_PALETTE[palette_index % len(KG_COLOR_PALETTE)]
                palette_index += 1
    
    # 添加节点，使用更现代的样式并增强交互体验
    for node in kg_data["nodes"]:
        node_id = node["id"]
        label = node.get("label", node_id)
        group = node.get("group", "Unknown")
        description = node.get("description", "")
        
        # 根据节点组类型设置颜色
        color = group_colors.get(group, KG_COLOR_PALETTE[0])  # 默认使用第一个颜色
        
        # 添加节点信息提示，改进格式
        title = f"{label}" + (f": {description}" if description else "")
        
        # 添加带有阴影和边框的节点 - 增加hover和select效果
        net.add_node(
            node_id, 
            label=label, 
            title=title, 
            color={
                "background": color, 
                "border": "#ffffff", 
                "highlight": {
                    "background": color, 
                    "border": "#000000"
                },
                "hover": {
                    "background": color, 
                    "border": "#000000"
                }
            }, 
            size=node_size, 
            font={"color": "#ffffff", "size": 14, "face": "Arial"},
            shadow={"enabled": True, "color": "rgba(0,0,0,0.2)", "size": 3},
            borderWidth=2,
            # 添加自定义数据用于交互
            group=group,
            description=description
        )
    
    # 添加边，使用更现代的样式并增强交互体验
    for link in kg_data["links"]:
        source = link["source"]
        target = link["target"]
        label = link.get("label", "")
        weight = link.get("weight", 1)
        
        # 根据权重设置线的粗细和不透明度
        width = edge_width * min(1 + (weight * 0.2), 3)
        
        # 使用弯曲的箭头和平滑的线条
        smooth = {"enabled": True, "type": "dynamic", "roundness": 0.5}
        
        title = label
        
        # 添加带有阴影的边 - 增加hover和select效果
        net.add_edge(
            source, 
            target, 
            title=title, 
            label=label, 
            width=width, 
            smooth=smooth,
            color={
                "color": "#999999", 
                "highlight": "#666666",
                "hover": "#666666"
            },
            shadow={"enabled": True, "color": "rgba(0,0,0,0.1)"},
            selectionWidth=2,
            # 添加自定义数据用于交互
            weight=weight,
            arrowStrikethrough=False
        )
    
    # 使用临时文件保存并显示网络图
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
            # 添加自定义样式和交互脚本
            # 导入样式
            from .kg_styles import KG_STYLES
            html_content = html_content.replace('</head>', KG_STYLES + '</head>')
            
            # 导入交互脚本
            from .interaction import KG_INTERACTION_SCRIPT
            html_content = html_content.replace('</body>', KG_INTERACTION_SCRIPT + '</body>')
            
            components.html(html_content, height=600)
        
        # 清理临时文件
        try:
            os.unlink(tmp.name)
        except:
            pass
    
    # 显示图例，使用更现代的样式
    st.write("### 图例")

    # 按特定优先级顺序显示图例
    priority_groups = ["Center", "Source", "Target", "Common"]
    community_groups = []
    other_groups = []

    # 对组类型进行分类
    for group in group_colors.keys():
        if group in priority_groups:
            continue  # 这些将单独处理
        elif isinstance(group, str) and "Community" in group:
            community_groups.append(group)
        else:
            other_groups.append(group)

    # 排序以确保一致的显示顺序
    community_groups.sort()
    other_groups.sort()

    # 合并所有组，保持优先顺序
    all_groups = []
    for group in priority_groups:
        if group in group_colors:
            all_groups.append(group)
    all_groups.extend(other_groups)
    all_groups.extend(community_groups)

    # 创建多列显示，使用更美观的图例样式
    cols = st.columns(3)
    for i, group in enumerate(all_groups):
        if group in group_colors:
            color = group_colors[group]
            col_idx = i % 3
            with cols[col_idx]:
                group_display_name = group
                if group == "Center":
                    group_display_name = "中心节点"
                elif group == "Source":
                    group_display_name = "源节点"
                elif group == "Target":
                    group_display_name = "目标节点"
                elif group == "Common":
                    group_display_name = "共同邻居"
                    
                st.markdown(
                    f'<div style="display:flex;align-items:center;margin-bottom:12px">'
                    f'<div style="width:20px;height:20px;border-radius:50%;background-color:{color};margin-right:10px;box-shadow:0 2px 4px rgba(0,0,0,0.1);"></div>'
                    f'<span style="font-family:sans-serif;color:#333;">{group_display_name}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )