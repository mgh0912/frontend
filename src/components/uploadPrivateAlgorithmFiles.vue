<template>
  <a-button  type="primary" ghost @click="uploadPrivateAlgorithmFiles()" v-if="!dialogVisible">上传私有算法</a-button>
  <a-form v-if="dialogVisible" :rules="rules">
    
    <div style="width: 200px">
      <span>算法类型：</span>
      <a-select v-model:value="form.algorithmType" placeholder="请选择算法类型" style="width: 130px">
        <a-select-option v-for="option in options" :key="option.value" :value="option.value">
          {{ option.label }}
        </a-select-option>
      </a-select>
    </div>
    
    
    <!-- 当上传故障诊断算法时，需要进一步选择是机器学习的还是深度学习的故障诊断 -->
    
    <div style="padding-top: 0;">
      <a-radio-group v-if="form.algorithmType === '故障诊断'" v-model:value="form.faultDiagnosisType" name="gradioGroup" style="width: 200px">
        <a-radio value="machineLearning" >机器学习</a-radio>
        <a-radio value="deepLearning" >深度学习</a-radio>
      </a-radio-group>
    </div>
    
    <div style="display: flex; justify-content: space-between;">
      <a-upload :file-list="fileList" :before-upload="beforeUpload" @remove="handleRemove">
        <a-button>
          <upload-outlined></upload-outlined>
          选择文件
        </a-button>
      </a-upload>

      <a-button
          type="primary"
          :disabled="fileList.length === 0"
          :loading="uploading"
          @click="handleUpload"
      >
        {{ uploading ? '上传中' : '开始上传' }}
      </a-button>
    </div>
  </a-form>

</template>

<script setup lang="ts">
import api from '../utils/api.js'
import { UploadOutlined } from '@ant-design/icons-vue';
import { message } from 'ant-design-vue';
import type { UploadProps } from 'ant-design-vue';
import { ref } from 'vue'
import {Action, ElMessageBox} from "element-plus";
import { useRouter } from "vue-router";
import {Rule} from "ant-design-vue/es/form";
const router = useRouter();
const fileList = ref<UploadProps['fileList']>([]);
const uploading = ref<boolean>(false);
const rules: Record<string, Rule[]> = {

  description: [
    { required: true, message: "请输入文件描述", trigger: "blur" },
  ],
}

const form = ref({
  algorithmType: '',
  fileList: [],
  faultDiagnosisType: 'machineLearning'
})

const options = ref([
    {value: '插值处理', label: '插值处理'},
    {value: '特征提取', label: '特征提取'},
    {value: '无量纲化', label: '无量纲化'},
    {value: '特征选择', label: '特征选择'},
    {value: '小波变换', label: '小波变换'},
    {value: '故障诊断', label: '故障诊断'},
    {value: '故障预测', label: '故障预测'},
    {value: '健康评估', label: '健康评估'},
])
const dialogVisible = ref(false)
const uploadPrivateAlgorithmFiles = () => {
  dialogVisible.value = true
}


// const faultDiagnosisType = ref('machineLearning')

const handleRemove: UploadProps['onRemove'] = file => {
  const index = fileList.value.indexOf(file);
  const newFileList = fileList.value.slice();
  newFileList.splice(index, 1);
  fileList.value = newFileList;
};

const beforeUpload: UploadProps['beforeUpload'] = file => {
  const isPythonFile = file.name.endsWith('.py');
  if (!isPythonFile) {
    message.error('只能上传 .py 文件');
    return false;
  }
  fileList.value = [...(fileList.value || []), file];
  return false;
};

const handleUpload = () => {
  if (!form.value.algorithmType) {
    message.error('请选择算法类型');
    return;
  }

  const formData = new FormData();
  // fileList.value.forEach((file: UploadProps['fileList'][number]) => {
  //   formData.append('files[]', file as any);
  // });
  formData.append("file", fileList.value[0]);
  formData.append("algorithm_type", form.value.algorithmType);
  formData.append("faultDiagnosisType", form.value.faultDiagnosisType);

  uploading.value = true;


  api.post("/user/upload_user_private_algorithm/", formData)
      .then((response: any) => {
        if (response.data.code == 200) {
          fileList.value = [];
          uploading.value = false;
          message.success("算法文件上传成功");
          dialogVisible.value = true;

        } else {
          uploading.value = false;
          message.error("算法文件上传失败, " + response.data.message);
        }
        if (response.data.code == 401) {
          ElMessageBox.alert('登录状态已失效，请重新登陆', '提示', {
            confirmButtonText: '确定',
            callback: (action: Action) => {
              router.push('/')
            },
          })
        }

      })
      .catch((error: any) => {
        uploading.value = false;
        message.error("上传失败, 请重试");
      });
}
</script>

<style scoped>

</style>