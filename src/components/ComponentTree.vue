<template>
  <div>
    <!-- 搜索框 -->
    <el-input
      v-model="searchKeywordOfTree"
      placeholder="请输入关键字查询"
      style="width: 100%; margin-bottom: 5px"
    />
    <!-- 新增树形结构 -->
    <div style="width: 100%; display: flex; flex-direction: row" v-if="props.userRole === 'superuser'">
      <el-input
        v-model="nameOfNewTree"
        placeholder="输入名称"
        style="width: 60%; margin-bottom: 5px"
      />
      <el-button style="width: 40%" @click="addNewComponentTree"
        >新增模型结构树</el-button
      >
    </div>
    <!-- 已有树形结构，使用 scoped-slot 渲染 -->
    <div style="width: 100%; overflow-x: auto">
      <div style="">
        <el-tree
          ref="treeRef"
          :data="filteredDataOfTree"
          style="width: 100%; max-width: 100%"
          node-key="id"
          :expand-on-click-node="false"
          :default-expand-all="isExpandAllOfTree"
          :filter-node-method="filterNodeOfTree"
          :accordion="false"
        >
          <template #default="{ data }">
            <span class="custom-tree-node" style="">
              <el-icon v-if="data.isModel" class="model-icon">
                <img src="@/assets/model-icon.svg" style="width: 20px; height: 20px;" alt="model">
              </el-icon>
              <span class="node-label">{{ getNodeLable(data) }}</span>
              <span class="node-actions" v-if="props.userRole === 'superuser' && !data.isModel">
                <el-icon @click="appendNodeOfTree(data)" :style="{ color: '#67c23a' }" v-if="data.disabled === true">
                  <Plus />
                </el-icon>
                <el-icon @click="removeNodeOfTree(data)" :style="{ color: '#f56c6c' }">
                  <Delete />
                </el-icon>
                <el-icon @click="editOfTree(data)" :style="{ color: '#409eff' }">
                  <Edit />
                </el-icon>
              </span>
            </span>
          </template>
        </el-tree>
        <!-- 编辑对话框 -->
        <el-dialog
          v-model="isEditDialogVisibleOfTree"
          title="编辑节点"
          :close-on-click-modal="false"
          :before-close="handleCloseOfTree"
          class="custom-dialog"
          style="width: 40%"
        >
          <span style="margin-right: 20px">新的节点名称：</span>
          <el-input
            v-model="editNodeLabelOfTree"
            placeholder="请输入新的节点名称"
            :autofocus="true"
            class="half-width-input"
            style="width: 50%"
          />
          <div style="font-size: 15px; color: gray">只能包含中英文、数字和下划线</div>
          <!-- 标记节点为最终类型后，不可添加子节点，而可以向其中添加模型 -->
          <span>是否标记节点为最终类型：</span>
          <el-radio-group v-model="isNodeEditable" style="width: 50%">
            <el-radio :value="true" size="large">否</el-radio>
            <el-radio :value="false" size="large">是</el-radio>
          </el-radio-group>
          <!-- <span slot="footer" class="dialog-footer">
                <el-button @click="isEditDialogVisibleOfTree = false">取消</el-button>
                <el-button type="primary" @click="saveEditOfTree">保存</el-button>
            </span> -->
          <template #footer>
            <div class="dialog-footer">
              <el-button @click="isEditDialogVisibleOfTree = false" style="width: 100px"
                >取消</el-button
              >
              <el-button type="primary" @click="saveEditOfTree" style="width: 100px"
                >确认</el-button
              >
            </div>
          </template>
        </el-dialog>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ElMessage, ElTree } from "element-plus";
import { ref, watch, reactive, onMounted } from "vue";
import api from "../utils/api.js";
import { isNode } from "@vue-flow/core";

interface Tree {
  label: string; // 节点名称
  value: string; // 节点id
  disabled: boolean; // 是否禁用添加子类型
  children?: Tree[]; // 子节点
  isModel: boolean;  // 是否为模型
  modelId: string;  // 模型id
  isPublished: string; // 是否发布
}

const treeRef = ref<InstanceType<typeof ElTree>>();

let idOfTree = 1000;

//是否展开全部
let isExpandAllOfTree = ref(false);

// 搜索关键字
const searchKeywordOfTree = ref("");
// 是否显示编辑对话框
const isEditDialogVisibleOfTree = ref(false);
// 当前编辑的节点
const editingNodeOfTree = ref<Tree | null>(null);
// 编辑框中的节点名称
const editNodeLabelOfTree = ref("");

// const dataSourceOfTree = reactive<Tree[]>([
//   {
//     value: 1,
//     label: "某型号轨道列车",
//     disabled: true, // 禁用该节点
//     children: [
//       {
//         value: 1.1,
//         label: "转向架",
//         disabled: true, // 禁用该节点
//         children: [
//           {
//             value: 1.1.1,
//             label: "构架系统",
//             disabled: true, // 禁用该节点
//             children: [
//               {
//                 value: 1.1.1.1,
//                 label: "构架组成",
//                 disabled: true, // 禁用该节点
//                 children: [
//                   { value: 1.1.1.1.1, label: "侧梁组成" },
//                   { value: 1.1.1.1.2, label: "空气弹簧座板" },
//                   { value: 1.1.1.1.3, label: "高度阀调整杆安装销" },
//                   { value: 11114, label: "扣板组成" },
//                   { value: 11115, label: "端梁座" },
//                   { value: 11116, label: "弹簧座" },
//                   { value: 11117, label: "弹簧座板" },
//                   { value: 11118, label: "闸线架支座" },
//                   { value: 11119, label: "线夹座" },
//                   { value: 11120, label: "止挡座组成" },
//                   { value: 11121, label: "安全吊链座" },
//                   { value: 11122, label: "侧梁工艺块" },
//                   { value: 11123, label: "长筋板" },
//                   { value: 11124, label: "托板组成" },
//                   { value: 11125, label: "短扣板" },
//                   { value: 11126, label: "压差阀座" },
//                   { value: 11127, label: "转向架铭牌" },
//                   { value: 11128, label: "构架序列号标志牌" },
//                   { value: 11129, label: "短筋板" },
//                 ],
//               },
//               {
//                 value: 1112,
//                 label: "构架衡量组成",
//                 disabled: true, // 禁用该节点
//                 children: [
//                   { value: 1113, label: "横梁钢管组成" },
//                   { value: 1114, label: "牵引电机吊座组成" },
//                   { value: 1115, label: "齿轮箱吊座组成" },
//                   { value: 1116, label: "构架牵引拉杆座" },
//                   { value: 1117, label: "纵向连接梁" },
//                   { value: 1118, label: "横向止挡组成" },
//                   { value: 1119, label: "横梁组成垂直挡" },
//                   { value: 1120, label: "构架外牵引拉杆座" },
//                 ],
//               },
//               {
//                 value: 1121,
//                 label: "构架端梁组成",
//                 disabled: true, // 禁用该节点
//                 children: [
//                   { value: 1122, label: "端梁安装梁" },
//                   { value: 1123, label: "端梁安装座组成" },
//                 ],
//               },
//             ],
//           },
//           {
//             value: 112,
//             label: "转向架附件",
//             disabled: true, // 禁用该节点
//             children: [
//               { value: 113, label: "转向架排障装置" },
//               { value: 114, label: "轮缘润滑装置" },
//             ],
//           },
//           {
//             value: 115,
//             label: "牵引装置系统",
//             disabled: true, // 禁用该节点
//             children: [
//               { value: 116, label: "中心销" },
//               { value: 117, label: "牵引梁组成" },
//               { value: 118, label: "牵引拉杆组成" },
//               { value: 119, label: "下盖" },
//             ],
//           },
//           // ... 其他子项
//         ],
//       },
//       {
//         value: 12,
//         label: "车体",
//         disabled: true, // 禁用该节点
//         children: [
//           { value: 121, label: "车顶" },
//           { value: 122, label: "车身侧面" },
//           { value: 123, label: "车身底部" },
//         ],
//       },
//       {
//         value: 13,
//         label: "电气系统",
//         disabled: true, // 禁用该节点
//         children: [
//           { value: 131, label: "牵引供电系统" },
//           { value: 132, label: "辅助供电系统" },
//           { value: 133, label: "列车控制系统" },
//         ],
//       },
//       {
//         value: 14,
//         label: "车内设施",
//         disabled: true, // 禁用该节点
//         children: [
//           { value: 141, label: "座椅和客舱布局" },
//           { value: 142, label: "餐饮和服务设施" },
//           { value: 143, label: "信息显示和广播系统" },
//         ],
//       },
//     ],
//   },
// ]);

// 定义从父组件接收到的userrole
const props = defineProps({
  userRole: String,
});

// 树形结构数据
const dataSourceOfTree = reactive<Tree[]>([]);
onMounted(() => {
  // 获取树形结构
  getComponentTrees();
});

// 从后端获取树形结构
const getComponentTrees = () => {
  api.get("user/get_component_trees").then((response: any) => {
    // 请求成功，后端返回数据
    if (response.data.code === 200) {
      dataSourceOfTree.length = 0;
      response.data.trees.map((tree: Tree) => dataSourceOfTree.push(tree));

      console.log("获取到树形结构: ", dataSourceOfTree);
    } else {
      ElMessage.error("获取树形结构失败，" + response.data.message);
    }
  });
};

const nameOfNewTree = ref("");
// 新增组件树形结构
const addNewComponentTree = () => {
  // 检验nameOfNewTree的格式，不能为空，且只能包含中英文、下划线和数字
  if (nameOfNewTree.value.trim() === "") {
    ElMessage.error("请输入组件名称");
    return;
  }
  if (!/^[a-zA-Z0-9_\u4e00-\u9fa5]+$/.test(nameOfNewTree.value)) {
    ElMessage.error("组件名称只能包含中英文、下划线和数字");
    return;
  }
  // 向后端发送建立新的结构树的post请求，请求内容包含新的结构树的树名
  let formData = new FormData();
  formData.append("treeName", nameOfNewTree.value);
  api
    .post("user/create_component_tree", formData)
    .then((response: any) => {
      // 请求成功，后端返回数据
      if (response.data.code === 200) {
        // 添加成功
        ElMessage.success("添加成功");
        getComponentTrees();
        // 重置输入框
        nameOfNewTree.value = "";
      } else {
        ElMessage.error("新建树失败，" + response.data.message);
      }
    })
    .catch((error: any) => {
      console.log(error);
      ElMessage.error("新建树失败，" + error);
    });
};

// 根据子节点的value 获取根节点的value（树型结构的名称）
// const getParentValue = (value: string) => {
//     let parentValue = "";
//     const parentValue = dataSourceOfTree.find((tree: Tree) => {

//     });
//     return parentValue ? parentValue.value : "";
// };

// 搜索过滤后的数据
const filteredDataOfTree = ref<Tree[]>(dataSourceOfTree);

// 添加子节点
const appendNodeOfTree = (data: Tree) => {
  console.log("添加节点方法执行...");
  console.log("data: ", data);

  if (data.disabled) {
    let parentNodeValue = data.value;
    let treeName = data.value.split(".")[0];
    let formData = new FormData();
    formData.append("treeName", treeName);
    formData.append("parentNodeValue", parentNodeValue);

    api
      .post("user/add_node_to_tree/", formData)
      .then((response: any) => {
        if (response.data.code === 200) {
          ElMessage.success("添加节点成功");
          getComponentTrees();
        } else {
          ElMessage.error("添加节点失败, " + response.data.message);
        }
      })
      .catch((error: any) => {
        ElMessage.error(error + ", 添加节点失败, 请重试");
      });
  } else {
    ElMessage.warning("该节点不可添加子节点");
  }

  //   const newChild = { id: idOfTree++, label: "New Node", children: [] };
  //   if (!data.children) {
  //     data.children = [];
  //   }
  //   data.children.push(newChild);
  data.children = [];
};

// 删除节点
const removeNodeOfTree = (data: Tree) => {
  console.log("删除节点方法执行...");
  let formData = new FormData();
  formData.append("treeName", data.value.split(".")[0]);
  formData.append("nodeValue", String(data.value));

  api
    .post("user/delete_node/", formData)
    .then((response: any) => {
      if (response.data.code == 200) {
        ElMessage.success("节点删除成功");
        // 刷新数据
        getComponentTrees();
      } else {
        ElMessage.error("节点删除失败, " + response.data.message);
      }
    })
    .catch((error: any) => {
      ElMessage.error("节点删除失败, " + error);
    });
  //   console.log(node);

  //   const parent = node.parent;
  //   const children: Tree[] = parent.data.children || parent.data;
  //   const index = children.findIndex((d) => d.id === data.id);
  //   if (index !== -1) {
  //     children.splice(index, 1);
  //   }
};


const getNodeLable = (data: Tree) => {
  if (!data.isModel){
    return data.label
  }else{
    return data.isPublished == '已发布' ? data.label + " (已发布)" : data.label + " (未发布)";
  }
};

// 编辑节点
const editOfTree = (data: Tree) => {
  //   console.log("编辑节点方法执行...");
  //   console.log("editOfTree: ", data);

  //   let formData = new FormData();
  //   formData.append("treeName", data.nodeValue.split('.')[0]);
  //   formData.append("newNodeName", )

  editingNodeOfTree.value = data; // 正在修改的节点
  isEditDialogVisibleOfTree.value = true;

  editNodeLabelOfTree.value = data.label
  isNodeEditable.value = data.disabled
};
// 修改节点的可修改性
const isNodeEditable = ref(true);

// 保存编辑
const saveEditOfTree = () => {
  //   if (editingNodeOfTree.value) {
  //     editingNodeOfTree.value.label = editNodeLabelOfTree.value;
  //     let formData

  //     isEditDialogVisibleOfTree.value = false;
  //   }
  // 检查editNodeLabelOfTree的值不能为空，且只能包含为中英文、数字和下划线的组合
  //   console.log("saveEditOfTree data: ", data)

  if (
    !editNodeLabelOfTree.value ||
    !/^[a-zA-Z0-9_\u4e00-\u9fa5]+$/.test(editNodeLabelOfTree.value)
  ) {
    // console.log("data: ", data);
    ElMessage({
      message: "请输入有效的组件名称",
      type: "error",
    });
    return;
  } else {
    let formData = new FormData();
    if (editingNodeOfTree.value) {
      if(!isNodeEditable.value && editingNodeOfTree.value.children?.length){
        ElMessage.warning("无法将已经添加子节点的节点设置为最终类型节点")
        return
      }
      if (isNodeEditable.value && !editingNodeOfTree.value.disabled && editingNodeOfTree.value.children?.length ){
        ElMessage.warning("该节点类型下已经添加了模型，无法将其设为非最终类型节点")
        return
      }
      formData.append("treeName", editingNodeOfTree.value.value.split(".")[0]); // 树的根节点的名称
      formData.append("newNodeName", editNodeLabelOfTree.value); // 新节点的名称
      formData.append("nodeValue", editingNodeOfTree.value.value);
      formData.append("nodeType", isNodeEditable.value ? "node" : "leaf");
    } else {
      console.log("editingNodeOfTree is null");
      return;
    }

    api.post("user/edit_node_name/", formData).then((response: any) => {
      if (response.data.code === 200) {
        ElMessage({
          message: "修改成功",
          type: "success",
        });
        // 刷新树
        getComponentTrees();

        isEditDialogVisibleOfTree.value = false;
      } else {
        ElMessage({
          message: "修改失败，" + response.data.message,
        });
      }
    });
  }
};

// 关闭编辑对话框时清理状态
const handleCloseOfTree = () => {
  editingNodeOfTree.value = null;
  editNodeLabelOfTree.value = "";
};

// 递归过滤节点数据
const filterDataOfTree = (nodes: Tree[], keyword: string): Tree[] => {
  return nodes
    .filter((node) => node.label.toLowerCase().includes(keyword.toLowerCase())) // 匹配节点名称
    .map((node) => {
      // 递归过滤子节点
      if (node.children) {
        node.children = filterDataOfTree(node.children, keyword);
      }
      return node;
    });
};

// 处理搜索输入
watch(searchKeywordOfTree, (val) => {
  treeRef.value!.filter(val);
});

// const handleSearchOfTree = () => {
//   if (searchKeywordOfTree.value != '') {
//     filteredDataOfTree.value = filterDataOfTree(dataSourceOfTree.value, searchKeywordOfTree.value)
//   } else {
//     filteredDataOfTree.value = dataSourceOfTree.value  // 清空搜索时恢复原始数据
//   }
// }

// const filterNodeOfTree = (value: string, data: Tree) => {
//   if (!value) return true;
//   return data.label.includes(value);
// };
// 修改 filterNodeOfTree 方法的参数类型为 TreeNodeData
const filterNodeOfTree = (value: string, data: any) => {
  if (!value) return true;
  return data.label ? data.label.includes(value) : false;
};
</script>

<style scoped>
.custom-dialog {
  width: 30%;
  display: flex;
  flex-direction: column;
}

.custom-dialog .dialog-footer {
  width: 100%;
  display: flex;
  flex-direction: row;
  justify-content: right;
  align-items: flex-end;
}

.custom-dialog .dialog-footer .small-button {
  margin-left: 10px;
  width: 100px;
  padding: 8px 16px; /* Adjust padding to make buttons smaller */
}

.custom-tree-node {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 14px;
}

.node-actions {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-left: 8px;
}

.model-icon {
  width: 20px; 
  height: 20px;
}
</style>
