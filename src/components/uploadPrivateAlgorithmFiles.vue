<template>
  <a-button type="primary" ghost @click="uploadPrivateAlgorithmFiles()">上传私有算法</a-button>
  <a-modal v-model:open="dialogVisible" title="上传私有算法" 
  cancelText="取消" :ok-button-props="{ style:{display: 'none' } }" :cancel-button-props="{ style:{display: 'none' } }">
    <div style="display: flex; flex-direction: row; padding: 10px">
      <span>
        <div style="width: 200px; margin-bottom: 10px">
          <span>算法类型：</span>
          <a-select v-model:value="form.algorithmType" @change="selectAlgorithmType" placeholder="请选择算法类型" style="width: 130px">
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
        
        <div style="display: flex; flex-direction: column;">
          <a-upload style="width: 220px" :file-list="fileList" :before-upload="beforeUpload" @remove="handleRemove" :maxCount="fileCount">
            <a-button class="upload-button">
              <upload-outlined></upload-outlined>
              从本地选择私有算法文件
            </a-button>
          </a-upload>

          <a-button
            type="primary"
            :disabled="fileList?.length === 0"
            :loading="uploading"
            @click="handleUpload"
            class="upload-button"
          >
            {{ uploading ? '上传中' : '开始上传' }}
          </a-button>
        </div>
      </span>

      <span>
        <p style="font-size: 15px; font-weight: bold;">私有算法模版参考</p>
        <div> 
          <a-button 
          v-for="algorithm in privateAlgorithms" 
          @click="setAlgorithmTemplate(algorithm)"
          type="default"
          >{{ algorithm }}</a-button> 
        </div>
      </span>
      
    </div>
  </a-modal>

  <!-- 私有算法参考模版 -->
  <a-modal v-model:open="templateDialog" :width="700" :title="templateName+'参考模版'" :ok-button-props="{ style:{display: 'none' } }" :cancel-button-props="{ style:{display: 'none' } }">
    <el-scrollbar :height="600">
      <div v-if="templateName === '插值处理'" >
        <div>
          <h1>基本结构</h1>
          <h2>1. 数据输入</h2>
          <h3>插值处理的私有算法作为脚本运行时，需要从主程序获取两个参数：</h3>
          <h3>(1) --raw-data-filepath， 需要插值的原数据的存放路径</h3>
          <h3>(2) --interpolated-data-filepath， 插值后的结果数据的存放路径</h3>
          <h3>数据输入方法：通过python中的argparse库进行命令行参数的添加和解析</h3>
          <a-image :width="500" src="src/assets/interpolation-params.png" />
          <h3>上图为获取输入数据的示例。在获取到原数据的存放路径后，通过numpy中的load()读取原数据，即可使用相应的插值算法对其进行插值处理。</h3>
          <h2>2. 数据输出</h2>
          <h3>在完成插值处理后，使用numpy库，将对于原数据的插值结果保存到'interpolated_data_filepath'指出的存放路径</h3>
          <a-image :width="500" src="src/assets/interpolation-output.png"></a-image>
          <h2>3. 私有算法代码模板示例</h2>
          <h3>插值处理的私有算法模板代码如下：</h3>
          <a-image :width="500" src="src/assets/interpolation-outline.png"></a-image>
          <h3>其中'linear_interpolatin_for_signal'为用户自定义的私有插值算法</h3>
        </div>
      </div>
      <div v-if="templateName === '故障诊断'" >
        <div>
          <h1>基本结构</h1>
          <h2>1. 数据输入</h2>
          <h3>故障诊断的私有算法作为脚本运行时，需要从主程序获取两个参数：</h3>
          <h3>(1) --model-filepath， 故障诊断算法所使用模型的模型参数的存放路径</h3>
          <h3>(2) --input-filepath， 样本数据的存放路径</h3>
          <h3>数据输入方法：通过python中的argparse库进行命令行参数的添加和解析</h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-model-params.png" />
          <!-- <h3>在获取到原数据的存放路径后，通过numpy中的load()读取原数据，即可使用相应的插值算法对其进行插值处理。</h3> -->
          <h2>2. 模型加载</h2>
          <h3>在获得模型参数的存放路径之后，通过该路径加载模型参数以初始化模型</h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-load-model.png" />
          <h3>其中注意，如果需要使用gpu辅助计算，则需要指定gpu为gpu:0</h3>
          <h3></h3>
          <p style="font-size: 15px; font-weight: 600;">
          以pytorch为例，通过如下代码指定使用第0块gpu：<br/>
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          </p>
          <!-- <code>
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          </code> -->
          <h2>3. 模型推理及输出结果</h2>
          <h3>初始化模型之后，即可使用样本数据进行模型推理</h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-prediction.png"></a-image>
          <h3>其中，进行模型推理之后，1代表有故障，0代表无故障，并需要将推理结果打印，以作为输出结果传回主程序</h3>
          <h2>4. 私有算法代码模板示例</h2>
          <h3>插值处理的私有算法模板代码(模型结构的定义在该代码段之前)如下：</h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-outline.png"></a-image>
          <h3>其中故障诊断的模型结构需要在该源文件之中定义</h3>
        </div>
      </div>
    </el-scrollbar>
    
  </a-modal>


</template>

<script setup lang="ts">
import { UploadOutlined } from '@ant-design/icons-vue';
import { message } from 'ant-design-vue';
import type { UploadProps } from 'ant-design-vue';
import { ref } from 'vue'
import {Action, ElMessageBox} from "element-plus";
import { useRouter } from "vue-router";
import api from '../utils/api.js';

// 私有算法
const privateAlgorithms = ['插值处理', '特征提取', '特征选择', '小波变换', '无量纲化', '故障诊断', '故障预测', '健康评估'];
const templateName = ref('');

// 查看私有算法模板
const templateDialog = ref(false);

const setAlgorithmTemplate = (algorithm: string) => {
  templateName.value = algorithm;
  templateDialog.value = true;

  // switch (algorithm) {
  //   case '插值处理':
  //     algorithmTemplate.value = 'interpolationTemplate';
  //     break;
  //   case '特征提取':
  //     algorithmTemplate.value = 'featureExtractionTemplate';
  //     break;
  //   case '无量纲化':
  //     algorithmTemplate.value = 'normalizationTemplate';
  //     break;
  //   case '特征选择':
  //     algorithmTemplate.value = 'featureSelectionTemplate';
  //     break;
  //   case '小波变换':
  //     algorithmTemplate.value = 'waveletTransformTemplate';
  //     break;
  //   case '故障诊断':
  //     algorithmTemplate.value = 'faultDiagnosisTemplate';
  //     break;
  //   case '故障预测':
  //     algorithmTemplate.value = 'faultPredictionTemplate';
  //     break;
  //   case '健康评估':
  //     algorithmTemplate.value = 'healthEvaluationTemplate';
  //     break;
  // }
}


const router = useRouter();
const fileList = ref<UploadProps['fileList']>([]);
const uploading = ref<boolean>(false);
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

  // 在删除文件列表中文件的同时，重新计算ruleOfDFA，以保证用户上传私有故障诊断算法时，同时包含用于故障诊断的模型以及模型参数文件。
  const isFaultDiagnosis = form.value.algorithmType === '故障诊断';
  const isPyFile = file.type === 'application/x-python-code' || file.name.endsWith('.py');
  const isPklFile = file.name.endsWith('.pkl');
  const isPthFile = file.name.endsWith('.pth');
  if (isFaultDiagnosis){
    if (isPyFile && ruleOfDFA > -1) ruleOfDFA -= 1
    if ((isPklFile||isPthFile) && ruleOfDFA < 1) ruleOfDFA += 1
  }
  // 删除文件列表中用户上传的文件
  if (fileList.value){
    const index = fileList.value.indexOf(file);
    const newFileList = fileList.value.slice();
    newFileList.splice(index, 1);
    fileList.value = newFileList;
  }else{
    console.log('fileList is undefined');
  }
};


// 当用户选择算法类型时，清空文件列表
const selectAlgorithmType = () => {
  if (fileList.value){
    fileList.value.forEach(file => {
      handleRemove(file);
    });
  }
}

// const beforeUpload: UploadProps['beforeUpload'] = file => {
//   const isPythonFile = file.name.endsWith('.py');
//   if (!isPythonFile) {
//     message.error('只能上传 .py 文件');
//     return false;
//   }
//   fileList.value = [...(fileList.value || []), file];
//   return false;
// };


//根据用户选择上传的私有算法的类型设置文件上传数量
const fileCount = ref(1);

let ruleOfDFA = 0;
const beforeUpload = (file: any) => {
  const isFaultDiagnosis = form.value.algorithmType === '故障诊断';
  const isPyFile = file.type === 'application/x-python-code' || file.name.endsWith('.py');
  const isPklFile = file.name.endsWith('.pkl');
  const isPthFile = file.name.endsWith('.pth');

  let fileNum = fileList.value?.length;
  if (fileNum == 0) ruleOfDFA = 0;
  
  if (isFaultDiagnosis) {
    // if (fileNum > 1) {
    //   message.warn('上传私有故障诊断算法时，需要上传定义的故障诊断算法(.py文件)以及诊断时使用的模型的加载参数(.pkl或.pth文件)，共两个文件');
    //   return false;
    // }
  
    if (fileNum == 0 && !isPyFile && !isPklFile && !isPthFile ){
      message.error('请上传包含该私有故障诊断算法(.py类型)或是模型参数(.pth或是.pkl类型)的文件');
      return false;
    }
    //判断是否是用户上传的私有故障诊断算法时是否同时上传了故障诊断的模型以及相关的模型参数文件（共两个文件）
    if (fileNum == 1) {
      if (!(isPyFile && (ruleOfDFA == -1)) && !((isPthFile || isPklFile) && (ruleOfDFA == 1))){
        message.error('上传私有故障诊断算法时，需要上传定义的故障诊断算法(.py文件)以及诊断时使用的模型的加载参数(.pkl或.pth文件)');
        return false;
      }
    }
  } else {
    if (fileNum == 0 && !isPyFile){
      message.error('请上传包含该私有算法的.py类型的文件');
      return false;
    }
    // 当选择上传除故障诊断算法的其他私有算法时，文件列表中最多只能有一个py文件
    if (fileNum == 1){
      if (isPyFile){
        handleRemove(file);
        fileList.value = [...fileList.value, file];  //将文件添加到fileList中
        return false;
      }
      message.warn('上传该类型私有算法时，最多只能上传一个.py类型的文件')
      return false;
    }
    // if (fileNum > 0 || !isPyFile){
    //   message.error('上传该类型私有算法时，最多只能上传一个包含该私有算法的.py类型的文件');
    //   return false;
    // }
    // if (!isPyFile) {
    //   message.error('只能上传 .py 文件');
    //   return false;
    // }
  }

  fileList.value = [...fileList.value, file];  //将文件添加到fileList中
  // 用于判断是否是用户上传的私有故障诊断算法时是否同时上传了故障诊断的模型以及相关的模型参数文件
  if (isFaultDiagnosis && isPyFile) {
    ruleOfDFA += 1;
  }

  if (isFaultDiagnosis && (isPklFile || isPthFile)){
    ruleOfDFA -= 1;
  }

  console.log('beforeUpload: ', ruleOfDFA)

  return false; // 阻止默认上传行为
};

const handleUpload = () => {
  if (!form.value.algorithmType) {
    message.error('请选择算法类型');
    return;
  }
  let isFaultDiagnosis = form.value.algorithmType;
  const formData = new FormData();
  // fileList.value.forEach((file: UploadProps['fileList'][number]) => {
  //   formData.append('files[]', file as any);
  // });
  let fileNum = fileList.value?.length;

  if(fileNum == 0){
    message.error('请选择上传包含私有算法的源文件');
    return;
  }
  
  if( fileNum == 1 ){
    let algorithmType = form.value.algorithmType;
    if(algorithmType != '故障诊断'){
      //上传私有非故障诊断算法
      if (fileList.value){
        let isPyFile = fileList.value[0].name.endsWith('.py');
        if(!isPyFile){
          message.error('请选择上传包含该私有算法的py源文件');
          return;
        }
        let file = fileList.value[0];
        formData.append("algorithmFile", file);
      }
    }else{
      //上传私有故障诊断算法
      if (fileList.value){
        let isPyFile = fileList.value[0].name.endsWith('.py');
        let isPklFile = fileList.value[0].name.endsWith('.pkl');
        let isPthFile = fileList.value[0].name.endsWith('.pth');
        if(isPyFile){
          message.error('上传私有故障诊断算法时，需上传模型参数文件(.pkl或.pth类型)');
          return;
        }
        if(isPklFile || isPthFile){
          message.error('上传私有故障诊断算法时，需上传定义的故障诊断算法(.py类型)')
          return;
        }
      }
    }
  }else{

    if (!isFaultDiagnosis){
      message.error("上传非故障诊断的私有算法时，仅需要上传一个包含该算法的py源文件");
      return;
    }
    //上传私有故障诊断算法
    let algorithm;
    let modelParams;
    if(fileList.value && fileNum){
      for (let i = 0; i < fileNum; i++){
        if (fileList.value[i].name.endsWith('.py')){
          formData.append("algorithmFile", fileList.value[i]);
          algorithm = fileList.value[i].name.split('.')[0];
        }else{
          formData.append("modelParamsFile", fileList.value[i]);
          modelParams = fileList.value[i].name.split('.')[0];
        }
      }
      if(algorithm !== modelParams){
        message.error('上传定义的故障诊断算法(.py文件)以及诊断时使用的模型的加载参数(.pkl或.pth文件)时，两个文件名称需要保持一致');
        return;
      }
    }
  }
  
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
          ruleOfDFA = 0;
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
.upload-button{
  width: 200px;
}
</style>