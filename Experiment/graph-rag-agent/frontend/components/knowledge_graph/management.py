import streamlit as st
import pandas as pd
from utils.api import (
    get_entity_types, get_relation_types, create_entity, create_relation, 
    update_entity, update_relation, delete_entity, delete_relation, 
    get_entities, get_relations
)

def display_kg_management_tab(tabs):
    """显示知识图谱管理标签页内容"""
    with tabs[3]:  # 添加为第四个标签页
        st.markdown('<div class="debug-header">知识图谱管理</div>', unsafe_allow_html=True)
        
        # 创建子标签页
        management_tabs = st.tabs(["实体管理", "关系管理"])
        
        # 实体管理标签页
        with management_tabs[0]:
            display_entity_management()
        
        # 关系管理标签页
        with management_tabs[1]:
            display_relation_management()

def display_entity_management():
    """显示实体管理界面"""
    st.subheader("实体管理")
    
    # 创建操作区域
    operation_tabs = st.tabs(["查询实体", "创建实体", "更新实体", "删除实体"])
    
    # 查询实体
    with operation_tabs[0]:
        st.markdown("#### 查询实体")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            search_term = st.text_input("实体名称/ID", key="entity_search_term", placeholder="输入实体名称或ID")
        
        with col2:
            # 防御性获取实体类型
            try:
                entity_types = get_entity_types() or []
            except Exception as e:
                st.error(f"获取实体类型失败: {str(e)}")
                entity_types = []
                
            selected_type = st.selectbox(
                "实体类型",
                options=["全部"] + entity_types,
                key="entity_search_type"
            )
        
        if st.button("查询", key="entity_search_button"):
            with st.spinner("正在查询实体..."):
                try:
                    # 构建过滤条件
                    filters = {}
                    if search_term:
                        filters["term"] = search_term
                    if selected_type and selected_type != "全部":
                        filters["type"] = selected_type
                    
                    # 调用API获取实体
                    entities = get_entities(filters)
                    
                    # 防御性检查返回值
                    if entities is not None and len(entities) > 0:
                        df = pd.DataFrame(entities)
                        st.dataframe(df, use_container_width=True)
                        st.success(f"找到 {len(entities)} 个实体")
                    else:
                        st.info("未找到匹配的实体")
                except Exception as e:
                    st.error(f"查询失败: {str(e)}")
    
    # 创建实体
    with operation_tabs[1]:
        st.markdown("#### 创建实体")
        
        # 防御性获取实体类型
        try:
            entity_types = get_entity_types() or []
        except Exception as e:
            st.error(f"获取实体类型失败: {str(e)}")
            entity_types = []
        
        # 创建表单
        with st.form("create_entity_form"):
            entity_id = st.text_input("实体ID *", placeholder="输入唯一实体ID")
            entity_name = st.text_input("实体名称 *", placeholder="输入实体名称")
            entity_type = st.selectbox("实体类型 *", options=entity_types) if entity_types else st.text_input("实体类型 *", placeholder="输入实体类型")
            entity_description = st.text_area("实体描述", placeholder="输入实体描述")
            
            # 添加自定义属性
            st.markdown("##### 自定义属性 (可选)")
            
            # 使用同一个键前缀为每个字段创建唯一的键
            prop_key_prefix = "create_entity_prop"
            property_keys = []
            property_values = []
            
            # 初始化时添加两个空属性字段
            for i in range(5):
                col1, col2 = st.columns([1, 2])
                with col1:
                    key = st.text_input(f"属性名称 {i+1}", key=f"{prop_key_prefix}_key_{i}")
                    property_keys.append(key)
                with col2:
                    value = st.text_input(f"属性值 {i+1}", key=f"{prop_key_prefix}_value_{i}")
                    property_values.append(value)
            
            submitted = st.form_submit_button("创建实体")
            
            if submitted:
                if not entity_id or not entity_name or not entity_type:
                    st.error("请填写必填字段: 实体ID、实体名称和实体类型")
                else:
                    # 构建自定义属性字典
                    properties = {}
                    for i in range(len(property_keys)):
                        if property_keys[i] and property_values[i]:  # 只添加有键和值的属性
                            properties[property_keys[i]] = property_values[i]
                    
                    # 构建实体数据
                    entity_data = {
                        "id": entity_id,
                        "name": entity_name,
                        "type": entity_type,
                        "description": entity_description,
                        "properties": properties
                    }
                    
                    try:
                        # 调用API创建实体
                        with st.spinner("正在创建实体..."):
                            result = create_entity(entity_data)
                            if result is not None and result.get("success", False):
                                st.success(f"成功创建实体: {entity_id}")
                            else:
                                error_message = "未知错误"
                                if result is not None:
                                    error_message = result.get("message", "未知错误")
                                st.error(f"创建失败: {error_message}")
                    except Exception as e:
                        st.error(f"创建失败: {str(e)}")
    
    # 更新实体
    with operation_tabs[2]:
        st.markdown("#### 更新实体")
        
        # 第一步：选择要更新的实体
        entity_id_to_update = st.text_input("输入要更新的实体ID", key="update_entity_id")
        
        if entity_id_to_update:
            lookup_button = st.button("查找实体", key="lookup_entity_button")
            
            if lookup_button or "entity_to_update" in st.session_state:
                try:
                    # 如果没有缓存或ID发生变化，重新查询
                    if ("entity_to_update" not in st.session_state or 
                        st.session_state.entity_to_update is None or 
                        st.session_state.entity_to_update.get("id") != entity_id_to_update):
                        with st.spinner("正在查找实体..."):
                            # 查询实体
                            entities = get_entities({"term": entity_id_to_update})
                            if entities is not None and len(entities) > 0:
                                # 找到精确匹配的实体
                                entity = next((e for e in entities if e.get("id") == entity_id_to_update), entities[0])
                                st.session_state.entity_to_update = entity
                            else:
                                st.error(f"未找到ID为 {entity_id_to_update} 的实体")
                                if "entity_to_update" in st.session_state:
                                    del st.session_state.entity_to_update
                                return
                    
                    # 显示实体更新表单
                    if "entity_to_update" in st.session_state and st.session_state.entity_to_update is not None:
                        entity = st.session_state.entity_to_update
                        
                        with st.form("update_entity_form"):
                            st.markdown(f"##### 更新实体: {entity.get('id')}")
                            
                            # 不允许更新ID
                            st.info(f"实体ID: {entity.get('id')} (不可更改)")
                            
                            # 获取当前值
                            current_name = entity.get("name", "")
                            current_type = entity.get("type", "")
                            current_description = entity.get("description", "")
                            current_properties = entity.get("properties", {}) or {}
                            
                            # 防御性获取实体类型
                            try:
                                entity_types_list = get_entity_types() or []
                                type_index = entity_types_list.index(current_type) if current_type in entity_types_list else 0
                            except Exception as e:
                                st.warning(f"获取实体类型失败，使用当前类型: {str(e)}")
                                entity_types_list = [current_type] if current_type else ["Unknown"]
                                type_index = 0
                            
                            # 表单字段
                            new_name = st.text_input("实体名称", value=current_name)
                            new_type = st.selectbox("实体类型", options=entity_types_list, index=type_index)
                            new_description = st.text_area("实体描述", value=current_description)
                            
                            # 属性编辑
                            st.markdown("##### 编辑属性")
                            
                            # 现有属性
                            st.markdown("现有属性:")
                            prop_updates = {}
                            prop_to_delete = []
                            
                            for key, value in current_properties.items():
                                col1, col2, col3 = st.columns([1, 2, 0.5])
                                with col1:
                                    st.text(key)
                                with col2:
                                    new_value = st.text_input(f"值: {key}", value=value, key=f"prop_{key}")
                                    prop_updates[key] = new_value
                                with col3:
                                    if st.checkbox("删除", key=f"delete_prop_{key}"):
                                        prop_to_delete.append(key)
                            
                            # 新属性
                            st.markdown("添加新属性:")
                            new_props = {}
                            
                            for i in range(3):  # 允许添加3个新属性
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    new_key = st.text_input(f"新属性名 {i+1}", key=f"new_prop_key_{i}")
                                with col2:
                                    new_val = st.text_input(f"新属性值 {i+1}", key=f"new_prop_val_{i}")
                                
                                if new_key and new_val:
                                    new_props[new_key] = new_val
                            
                            submitted = st.form_submit_button("更新实体")
                            
                            if submitted:
                                try:
                                    # 构建更新数据
                                    update_data = {
                                        "id": entity.get("id"),
                                        "name": new_name,
                                        "type": new_type,
                                        "description": new_description,
                                    }
                                    
                                    # 处理属性更新
                                    properties = {**current_properties}  # 复制当前属性
                                    for key in prop_to_delete:  # 删除标记的属性
                                        if key in properties:
                                            del properties[key]
                                    
                                    # 更新属性值
                                    for key, value in prop_updates.items():
                                        if key not in prop_to_delete:  # 不更新将被删除的属性
                                            properties[key] = value
                                    
                                    # 添加新属性
                                    properties.update(new_props)
                                    
                                    update_data["properties"] = properties
                                    
                                    # 调用API更新实体
                                    with st.spinner("正在更新实体..."):
                                        result = update_entity(update_data)
                                        if result is not None and result.get("success", False):
                                            st.success(f"成功更新实体: {entity.get('id')}")
                                            # 更新缓存的实体
                                            st.session_state.entity_to_update = {**update_data}
                                        else:
                                            error_message = "未知错误"
                                            if result is not None:
                                                error_message = result.get("message", "未知错误")
                                            st.error(f"更新失败: {error_message}")
                                except Exception as e:
                                    st.error(f"更新失败: {str(e)}")
                
                except Exception as e:
                    st.error(f"查找实体时出错: {str(e)}")
    
    # 删除实体
    with operation_tabs[3]:
        st.markdown("#### 删除实体")
        
        delete_id = st.text_input("输入要删除的实体ID", key="delete_entity_id")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            confirm = st.checkbox("确认删除", key="confirm_entity_delete")
        
        with col2:
            if delete_id and confirm:
                if st.button("删除实体", key="delete_entity_button"):
                    try:
                        with st.spinner("正在删除实体..."):
                            result = delete_entity(delete_id)
                            if result is not None and result.get("success", False):
                                st.success(f"成功删除实体: {delete_id}")
                                # 如果缓存中有此实体，清除缓存
                                if ("entity_to_update" in st.session_state and 
                                    st.session_state.entity_to_update is not None and 
                                    st.session_state.entity_to_update.get("id") == delete_id):
                                    del st.session_state.entity_to_update
                            else:
                                error_message = "未知错误"
                                if result is not None:
                                    error_message = result.get("message", "未知错误")
                                st.error(f"删除失败: {error_message}")
                    except Exception as e:
                        st.error(f"删除失败: {str(e)}")
            else:
                st.info("请输入实体ID并确认删除")

def display_relation_management():
    """显示关系管理界面"""
    st.subheader("关系管理")
    
    # 创建操作区域
    operation_tabs = st.tabs(["查询关系", "创建关系", "更新关系", "删除关系"])
    
    # 查询关系
    with operation_tabs[0]:
        st.markdown("#### 查询关系")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            source_entity = st.text_input("源实体ID", key="relation_search_source", placeholder="源实体ID (可选)")
        
        with col2:
            target_entity = st.text_input("目标实体ID", key="relation_search_target", placeholder="目标实体ID (可选)")
        
        with col3:
            # 防御性获取关系类型
            try:
                relation_types = get_relation_types() or []
            except Exception as e:
                st.error(f"获取关系类型失败: {str(e)}")
                relation_types = []
                
            selected_rel_type = st.selectbox(
                "关系类型",
                options=["全部"] + relation_types,
                key="relation_search_type"
            )
        
        if st.button("查询", key="relation_search_button"):
            with st.spinner("正在查询关系..."):
                try:
                    # 构建过滤条件
                    filters = {}
                    if source_entity:
                        filters["source"] = source_entity
                    if target_entity:
                        filters["target"] = target_entity
                    if selected_rel_type and selected_rel_type != "全部":
                        filters["type"] = selected_rel_type
                    
                    # 调用API获取关系
                    relations = get_relations(filters)
                    
                    # 防御性检查返回值
                    if relations is not None and len(relations) > 0:
                        df = pd.DataFrame(relations)
                        st.dataframe(df, use_container_width=True)
                        st.success(f"找到 {len(relations)} 条关系")
                    else:
                        st.info("未找到匹配的关系")
                except Exception as e:
                    st.error(f"查询失败: {str(e)}")
    
    # 创建关系
    with operation_tabs[1]:
        st.markdown("#### 创建关系")
        
        # 防御性获取关系类型
        try:
            relation_types = get_relation_types() or []
        except Exception as e:
            st.error(f"获取关系类型失败: {str(e)}")
            relation_types = []
        
        # 创建表单
        with st.form("create_relation_form"):
            source_id = st.text_input("源实体ID *", placeholder="输入源实体ID")
            relation_type = st.selectbox("关系类型 *", options=relation_types) if relation_types else st.text_input("关系类型 *", placeholder="输入关系类型")
            target_id = st.text_input("目标实体ID *", placeholder="输入目标实体ID")
            
            relation_description = st.text_area("关系描述", placeholder="输入关系描述")
            
            weight = st.slider("关系权重", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            
            # 添加自定义属性
            st.markdown("##### 自定义属性 (可选)")
            
            prop_key_prefix = "create_relation_prop"
            property_keys = []
            property_values = []
            
            # 初始化时添加三个空属性字段
            for i in range(3):
                col1, col2 = st.columns([1, 2])
                with col1:
                    key = st.text_input(f"属性名称 {i+1}", key=f"{prop_key_prefix}_key_{i}")
                    property_keys.append(key)
                with col2:
                    value = st.text_input(f"属性值 {i+1}", key=f"{prop_key_prefix}_value_{i}")
                    property_values.append(value)
            
            submitted = st.form_submit_button("创建关系")
            
            if submitted:
                if not source_id or not target_id or not relation_type:
                    st.error("请填写必填字段: 源实体ID、关系类型和目标实体ID")
                else:
                    # 构建自定义属性字典
                    properties = {}
                    for i in range(len(property_keys)):
                        if property_keys[i] and property_values[i]:  # 只添加有键和值的属性
                            properties[property_keys[i]] = property_values[i]
                    
                    # 构建关系数据
                    relation_data = {
                        "source": source_id,
                        "type": relation_type,
                        "target": target_id,
                        "description": relation_description,
                        "weight": weight,
                        "properties": properties
                    }
                    
                    try:
                        # 调用API创建关系
                        with st.spinner("正在创建关系..."):
                            result = create_relation(relation_data)
                            if result is not None and result.get("success", False):
                                st.success(f"成功创建关系: {source_id} -[{relation_type}]-> {target_id}")
                            else:
                                error_message = "未知错误"
                                if result is not None:
                                    error_message = result.get("message", "未知错误")
                                st.error(f"创建失败: {error_message}")
                    except Exception as e:
                        st.error(f"创建失败: {str(e)}")
    
    # 更新关系
    with operation_tabs[2]:
        st.markdown("#### 更新关系")
        
        # 选择要更新的关系
        st.markdown("##### 查找要更新的关系")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            source_filter = st.text_input("源实体ID", key="update_relation_source")
        
        with col2:
            # 防御性获取关系类型
            try:
                relation_types_list = get_relation_types() or []
            except Exception as e:
                st.error(f"获取关系类型失败: {str(e)}")
                relation_types_list = []
                
            relation_type_filter = st.selectbox(
                "关系类型",
                options=["全部"] + relation_types_list,
                key="update_relation_type"
            )
        
        with col3:
            target_filter = st.text_input("目标实体ID", key="update_relation_target")
        
        lookup_button = st.button("查找关系", key="lookup_relation_button")
        
        if lookup_button:
            try:
                with st.spinner("正在查找关系..."):
                    # 构建过滤条件
                    filters = {}
                    if source_filter:
                        filters["source"] = source_filter
                    if target_filter:
                        filters["target"] = target_filter
                    if relation_type_filter and relation_type_filter != "全部":
                        filters["type"] = relation_type_filter
                    
                    # 调用API获取关系
                    relations = get_relations(filters)
                    
                    if relations is not None and len(relations) > 0:
                        # 存储找到的关系
                        st.session_state.found_relations = relations
                        
                        # 显示结果表格
                        df = pd.DataFrame(relations)
                        st.dataframe(df, use_container_width=True)
                        
                        # 选择要更新的关系
                        relation_ids = [f"{r.get('source')} -[{r.get('type')}]-> {r.get('target')}" for r in relations]
                        selected_relation = st.selectbox(
                            "选择要更新的关系",
                            options=relation_ids,
                            key="relation_to_update_id"
                        )
                        
                        if selected_relation:
                            # 从ID字符串中解析三部分
                            parts = selected_relation.split(" -[")
                            source = parts[0]
                            type_target = parts[1].split("]-> ")
                            rel_type = type_target[0]
                            target = type_target[1]
                            
                            # 找到选中的关系对象
                            relation = next((r for r in relations if r.get("source") == source and r.get("type") == rel_type and r.get("target") == target), None)
                            
                            if relation is not None:
                                st.session_state.relation_to_update = relation
                                
                                with st.form("update_relation_form"):
                                    st.markdown(f"##### 更新关系: {source} -[{rel_type}]-> {target}")
                                    
                                    # 防御性获取关系类型
                                    try:
                                        rel_types = get_relation_types() or []
                                        type_index = rel_types.index(rel_type) if rel_type in rel_types else 0
                                    except Exception as e:
                                        st.warning(f"获取关系类型失败，使用当前类型: {str(e)}")
                                        rel_types = [rel_type] if rel_type else ["Unknown"]
                                        type_index = 0
                                    
                                    # 关系类型
                                    new_type = st.selectbox(
                                        "关系类型",
                                        options=rel_types,
                                        index=type_index
                                    )
                                    
                                    # 关系描述
                                    current_description = relation.get("description", "")
                                    new_description = st.text_area("关系描述", value=current_description)
                                    
                                    # 关系权重
                                    current_weight = relation.get("weight", 0.5)
                                    new_weight = st.slider("关系权重", min_value=0.0, max_value=1.0, value=current_weight, step=0.01)
                                    
                                    # 属性编辑
                                    st.markdown("##### 编辑属性")
                                    
                                    # 现有属性
                                    current_properties = relation.get("properties", {}) or {}
                                    st.markdown("现有属性:")
                                    prop_updates = {}
                                    prop_to_delete = []
                                    
                                    for key, value in current_properties.items():
                                        col1, col2, col3 = st.columns([1, 2, 0.5])
                                        with col1:
                                            st.text(key)
                                        with col2:
                                            new_value = st.text_input(f"值: {key}", value=value, key=f"rel_prop_{key}")
                                            prop_updates[key] = new_value
                                        with col3:
                                            if st.checkbox("删除", key=f"delete_rel_prop_{key}"):
                                                prop_to_delete.append(key)
                                    
                                    # 新属性
                                    st.markdown("添加新属性:")
                                    new_props = {}
                                    
                                    for i in range(2):
                                        col1, col2 = st.columns([1, 2])
                                        with col1:
                                            new_key = st.text_input(f"新属性名 {i+1}", key=f"new_rel_prop_key_{i}")
                                        with col2:
                                            new_val = st.text_input(f"新属性值 {i+1}", key=f"new_rel_prop_val_{i}")
                                        
                                        if new_key and new_val:
                                            new_props[new_key] = new_val
                                    
                                    submitted = st.form_submit_button("更新关系")
                                    
                                    if submitted:
                                        try:
                                            # 构建更新数据
                                            update_data = {
                                                "source": source,
                                                "original_type": rel_type,
                                                "target": target,
                                                "new_type": new_type,
                                                "description": new_description,
                                                "weight": new_weight
                                            }
                                            
                                            # 处理属性更新
                                            properties = {**current_properties}  # 复制当前属性
                                            for key in prop_to_delete:  # 删除标记的属性
                                                if key in properties:
                                                    del properties[key]
                                            
                                            # 更新属性值
                                            for key, value in prop_updates.items():
                                                if key not in prop_to_delete:  # 不更新将被删除的属性
                                                    properties[key] = value
                                            
                                            # 添加新属性
                                            properties.update(new_props)
                                            
                                            update_data["properties"] = properties
                                            
                                            # 调用API更新关系
                                            with st.spinner("正在更新关系..."):
                                                result = update_relation(update_data)
                                                if result is not None and result.get("success", False):
                                                    st.success(f"成功更新关系")
                                                    # 清除缓存的关系
                                                    if "relation_to_update" in st.session_state:
                                                        del st.session_state.relation_to_update
                                                else:
                                                    error_message = "未知错误"
                                                    if result is not None:
                                                        error_message = result.get("message", "未知错误")
                                                    st.error(f"更新失败: {error_message}")
                                        except Exception as e:
                                            st.error(f"更新失败: {str(e)}")
                    else:
                        st.warning("未找到匹配的关系")
            except Exception as e:
                st.error(f"查找关系时出错: {str(e)}")
    
    # 删除关系
    with operation_tabs[3]:
        st.markdown("#### 删除关系")
        
        st.markdown("##### 指定要删除的关系")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            delete_source = st.text_input("源实体ID *", key="delete_relation_source")
        
        with col2:
            # 防御性获取关系类型
            try:
                relation_types = get_relation_types() or []
            except Exception as e:
                st.error(f"获取关系类型失败: {str(e)}")
                relation_types = []
                
            delete_type = st.selectbox("关系类型 *", options=relation_types, key="delete_relation_type") if relation_types else st.text_input("关系类型 *", key="delete_relation_type_input")
        
        with col3:
            delete_target = st.text_input("目标实体ID *", key="delete_relation_target")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            confirm_delete = st.checkbox("确认删除", key="confirm_relation_delete")
        
        with col2:
            if delete_source and delete_type and delete_target and confirm_delete:
                if st.button("删除关系", key="delete_relation_button"):
                    try:
                        with st.spinner("正在删除关系..."):
                            # 构建删除数据
                            delete_data = {
                                "source": delete_source,
                                "type": delete_type,
                                "target": delete_target
                            }
                            
                            result = delete_relation(delete_data)
                            if result is not None and result.get("success", False):
                                st.success(f"成功删除关系: {delete_source} -[{delete_type}]-> {delete_target}")
                            else:
                                error_message = "未知错误"
                                if result is not None:
                                    error_message = result.get("message", "未知错误")
                                st.error(f"删除失败: {error_message}")
                    except Exception as e:
                        st.error(f"删除失败: {str(e)}")
            else:
                st.info("请完整填写关系信息并确认删除")