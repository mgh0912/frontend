<template>
  <a-button class="private-algorithm-button" ghost @click="uploadPrivateAlgorithmFiles()">
    <span>上传组件</span></a-button
  >
  <a-modal
    v-model:open="dialogVisible"
    title="上传增值服务组件"
    cancelText="取消"
    :ok-button-props="{ style: { display: 'none' } }"
    :cancel-button-props="{ style: { display: 'none' } }"
  >
    <div
      style="
        display: flex;
        padding: 10px;
        flex-direction: column;
        justify-content: center;
      "
    >
      <span style="width: 100%">
        <!-- 选择组件的算法类型 -->
        <div style="width: 100%; margin-bottom: 10px">
          <span><span style="color: red">*</span>选择上传组件的算法类型：</span>
          <!-- <a-select
            v-model:value="form.algorithmType"
            @change="selectAlgorithmType"
            placeholder="请选择算法类型"
            style="width: 150px"
          >
            <a-select-option
              v-for="option in options"
              :key="option.value"
              :value="option.value"
            >
              {{ option.label }}
            </a-select-option>
          </a-select> -->
          <a-tree-select
            v-model:value="form.algorithmType"
            show-search
            style="width: 40%"
            :dropdown-style="{ maxHeight: '400px', overflow: 'auto' }"
            placeholder="请选择要上传的组件类型"
            tree-default-expand-all
            :tree-data="treeData"
            tree-node-filter-prop="label"
            @change="algorithmTypeChange"
          >
            <template #title="{ value: val, label }">
              <b v-if="val === 'parent 1-1'" style="color: #08c">sss</b>
              <template v-else>{{ label }}</template>
            </template>
          </a-tree-select>
        </div>

        <!-- 当上传无量纲化算法时，需要进一步选择是提取特征还是原始信号进行无量纲化 -->
        <div
          style="padding-top: 0; padding-bottom: 10px"
          v-if="form.algorithmType === '无量纲化'"
        >
          <span><span style="color: red">*</span>选择无量纲化的对象：</span>
          <a-radio-group v-model:value="form.useLog" name="gradioGroup">
            <a-radio :value="false">对原始信号无量纲化</a-radio>
            <a-radio :value="true">对提取的特征无量纲化</a-radio>
          </a-radio-group>
        </div>

        <!-- 上传增值组件 -->
        <div style="display: flex; flex-direction: column; margin-top: 20px">
          <div style="display: flex; flex-direction: row">
            <!-- 上传增值服务的算法源文件 -->
            <div>
              <span><span style="color: red">*</span>选择要上传的算法源文件：</span>
              <a-upload
                :file-list="pythonFileList"
                :before-upload="beforeUploadAlgorithmFile"
                @remove="removePythonFile"
                :maxCount="1"
              >
                <a-button class="upload-button" :disabled="canSelectPythonFile">
                  <upload-outlined></upload-outlined>
                  从本地选择
                </a-button>
              </a-upload>
            </div>

            <!-- 一抽屉的形式弹出相关算法模板参考 -->
            <span>
              <a-tooltip title="算法模板参考">
                <a-button
                  type="default"
                  style="color: blue; margin-left: 10px"
                  @click="showDrawer = true"
                  shape="circle"
                  :icon="h(QuestionOutlined)"
                />
              </a-tooltip>
            </span>
          </div>

          <!-- 当上传故障诊断算法时，需要进一步选择是机器学习的还是深度学习的故障诊断 -->
          <div style="margin-top: 20px" v-if="form.algorithmType === '故障诊断'">
            <span><span style="color: red">*</span>选择所使用的模型类型：</span>
            <a-radio-group v-model:value="form.faultDiagnosisType" name="gradioGroup" @change="removeModelFile">
              <a-radio value="machineLearning">机器学习模型</a-radio>
              <a-radio value="deepLearning">深度学习模型</a-radio>
            </a-radio-group>
          </div>

          <!-- 上传算法所需的模型文件 -->
          <div
            style="display: flex; flex-direction: row; margin-top: 20px"
            v-if="
              canUploadModelFile || (form.algorithmType === '无量纲化' && form.useLog)
            "
          >
            <span><span style="color: red">*</span>选择所要使用的模型文件：</span>
            <a-upload
              :file-list="modelFileList"
              :before-upload="beforeUploadModelFile"
              @remove="removeModelFile"
              :maxCount="1"
            >
              <a-button class="upload-button">
                <upload-outlined></upload-outlined>
                从本地选择
              </a-button>
            </a-upload>
          </div>
        </div>
        <a-button
          type="primary"
          :disabled="(canUploadModelFile && (modelFileList?.length === 0 || pythonFileList?.length === 0)) || (!canUploadModelFile && pythonFileList?.length === 0)"
          :loading="uploading"
          @click="uploadExtraModule"
          class="upload-button"
          style="width: 160px; margin-left: 140px; margin-top: 30px"
        >
          {{ uploading ? "上传中" : "开始上传增值组件" }}
        </a-button>
      </span>

      <!-- 点击相应按钮时，显示私有算法模版参考 -->
      <!-- <a-button @click="showDrawer = true">显示算法模板参考</a-button> -->
      <a-drawer :visible="showDrawer" @close="onClose" placement="right" :closable="true">
        <span>
          <p style="font-size: 15px; font-weight: bold">上传算法模板参考</p>
          <div>
            <a-button
              v-for="algorithm in privateAlgorithms"
              @click="setAlgorithmTemplate(algorithm)"
              type="default"
            >
              {{ algorithm }}
            </a-button>
          </div>
        </span>
      </a-drawer>
      <!-- <span>
        <p style="font-size: 15px; font-weight: bold">上传算法模板参考</p>
        <div>
          <a-button
            v-for="algorithm in privateAlgorithms"
            @click="setAlgorithmTemplate(algorithm)"
            type="default"
            >{{ algorithm }}
          </a-button>
        </div>
      </span> -->
    </div>

    <!-- 管理 -->
    <div></div>
  </a-modal>

  <!-- 私有算法参考模版 -->
  <a-modal
    v-model:open="templateDialog"
    :width="700"
    :title="templateName + '参考模版'"
    :ok-button-props="{ style: { display: 'none' } }"
    :cancel-button-props="{ style: { display: 'none' } }"
  >
    <el-scrollbar :height="600">
      <div v-if="templateName === '插值处理'">
        <div>
          <h1>基本结构</h1>
          <h2>1. 数据输入</h2>
          <h3>插值处理的私有算法作为脚本运行时，需要从主程序获取两个参数：</h3>
          <h3>(1) --raw-data-filepath， 需要插值的原数据的存放路径</h3>
          <h3>(2) --interpolated-data-filepath， 插值后的结果数据的存放路径</h3>
          <h3>数据输入方法：通过python中的argparse库进行命令行参数的添加和解析</h3>
          <a-image :width="500" src="src/assets/interpolation-params.png" />
          <h3>
            上图为获取输入数据的示例。在获取到原数据的存放路径后，通过numpy中的load()读取原数据，即可使用相应的插值算法对其进行插值处理。
          </h3>
          <h2>2. 数据输出</h2>
          <h3>
            在完成插值处理后，使用numpy库，将对于原数据的插值结果保存到'interpolated_data_filepath'指出的存放路径
          </h3>
          <a-image :width="500" src="src/assets/interpolation-output.png"></a-image>
          <h2>3. 私有算法代码模板示例</h2>
          <h3>插值处理的私有算法模板代码如下：</h3>
          <a-image :width="500" src="src/assets/interpolation-outline.png"></a-image>
          <h3>其中'linear_interpolatin_for_signal'为用户自定义的私有插值算法</h3>
        </div>
      </div>
      <div v-if="templateName === '故障诊断'">
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
          <h3>机器学习和深度学习的模型加载方式不同</h3>
          <h3>
            (1)对于深度学习模型，在获得模型参数的存放路径之后，通过该路径加载模型参数以初始化模型，以pytorch框架为例，通过如下方式加载模型参数
          </h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-load-model.png" />
          <h3>
            (2)对于机器学习模型，在获得模型的存放路径之后，通过joblib库的'load()'方法加载已训练好的模型
          </h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-load-model-2.png" />
          <h3>其中注意，如果需要使用gpu辅助计算，则需要指定gpu为gpu:0</h3>
          <p style="font-size: 15px; font-weight: 600">
            以pytorch为例，通过如下代码指定使用第0块gpu：<br />
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          </p>
          <!-- <code>
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          </code> -->
          <h2>3. 模型推理及输出结果</h2>
          <h3>
            初始化模型之后，即可使用样本数据进行模型推理，进行模型推理之后，1代表有故障，0代表无故障，
          </h3>
          <h3>在此应该注意，注意深度学习和机器学习的故障诊断输入样本数据不同：</h3>
          <h3>
            (1)对于深度学习，其输入样本为原始的振动信号，因此使用numpy库的'numpy.load()'函数读取输入数据
          </h3>
          <h3>
            (2)对于机器学习，其输入样本为提取的手工特征，主程序中保存为pickle对象，因此使用pickle库的'pickle.load()'函数读取输入数据
          </h3>
          <h3>
            以使用pytorch框架训练的深度学习故障诊断模型为例，给出故障诊断示例代码如下
          </h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-prediction.png"></a-image>
          <h3>在得到诊断结果之后，需要以打印的形式作为输出结果传回主程序</h3>
          <br />
          <h3>
            除此之外，对于机器学习的故障诊断，需要选定模型训练时的手工特征从输入数据中提取模型推理所需的特征样本，下图为机器学习的故障诊断推理示例
          </h3>
          <a-image
            :width="500"
            src="src/assets/fault-diagnosis-prediction-ml.png"
          ></a-image>
          <h3>代码中的choose_features即为该模型所需的手工特征</h3>
          <h2>4. 私有算法代码模板示例</h2>
          <h3>故障诊断的私有算法模板代码(模型结构的定义在该代码段之前)如下：</h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-outline.png"></a-image>
          <h3>其中故障诊断的模型结构需要在该python源文件之中定义</h3>
          <br />
          <h3>附：示例代码源文件下载链接（点击下载）</h3>
          <a
            href="src/assets/exampleCode/My-FD-Algorithm-1.py"
            download="example-fault-diagnosis.py"
            >深度学习故障诊断示例代码源文件</a
          >
        </div>
      </div>
      <div v-if="templateName === '故障预测'">
        <div>
          <h1>基本结构</h1>
          <h2>1. 数据输入</h2>
          <h3>故障预测的私有算法作为脚本运行时，需要从主程序获取两个参数：</h3>
          <h3>(1) --model-filepath， 故障预测算法所使用模型的模型参数的存放路径</h3>
          <h3>(2) --input-filepath， 样本数据的存放路径</h3>
          <h3>数据输入方法：通过python中的argparse库进行命令行参数的添加和解析</h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-model-params.png" />
          <!-- <h3>在获取到原数据的存放路径后，通过numpy中的load()读取原数据，即可使用相应的插值算法对其进行插值处理。</h3> -->
          <h2>2. 模型加载</h2>
          <h3>机器学习和深度学习的模型加载方式不同</h3>
          <h3>
            (1)如果是深度学习，在获得模型参数的存放路径之后，通过该路径加载模型参数以初始化模型，以pytorch框架为例，通过如下方式加载模型参数
          </h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-load-model.png" />
          <h3>
            (2)如果是机器学习，在获得模型的存放路径之后，通过joblib库的'load()'方法加载已训练好的模型
          </h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-load-model-2.png" />
          <h3>其中注意，如果需要使用gpu辅助计算，则需要指定gpu为gpu:0</h3>
          <h3></h3>
          <p style="font-size: 15px; font-weight: 600">
            以pytorch为例，通过如下代码指定使用第0块gpu：<br />
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          </p>
          <!-- <code>
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          </code> -->
          <h2>3. 模型推理及输出结果</h2>
          <h3>
            初始化模型之后，即可使用样本数据进行模型推理。这里注意深度学习和机器学习的样本数据不同
          </h3>
          <h3>
            (1)对于深度学习，其输入样本为原始的振动信号，因此使用numpy库的'numpy.load()'函数读取输入数据
          </h3>
          <h3>
            (2)对于机器学习，其输入样本为提取的手工特征，因此使用pickle库的'pickle.load()'函数读取输入数据
          </h3>
          <a-image :width="500" src="src/assets/fault-diagnosis-prediction.png"></a-image>
          <h3>
            其中，进行模型推理之后，得到预测的可能出现故障的时间，需要使用print()打印推理结果，以作为输出结果传回主程序，
          </h3>
          <h3>
            此时，需要注意的是在打印预测结果时，应该按照如下所示在预测结果之后紧跟'#'符号，以便于主程序读取输出的结果
          </h3>
          <a-image :width="500" src="src/assets/fault-prediction-output.png"></a-image>
          <h2>4. 私有算法代码模板示例</h2>
          <h3>故障预测的私有算法模板代码如下：</h3>
          <a-image :width="500" src="src/assets/fault-prediction-outline.png"></a-image>
        </div>
      </div>
      <!-- 无量纲化的专有算法模版 -->
      <div v-if="templateName === '无量纲化'">
        <div>
          <h1>基本结构</h1>
          <h2>1. 数据输入</h2>
          <h3>故障预测的私有算法作为脚本运行时，需要从主程序获取三个参数：</h3>
          <h3>(1) --model-filepath， 故障预测算法所使用模型的模型参数的存放路径</h3>
          <h3>(2) --input-filepath， 样本数据的存放路径</h3>
          <h3>(3) --output-filepath,，处理结果的存放路径</h3>
          <h3>数据输入方法：通过python中的argparse库进行命令行参数的添加和解析</h3>
          <a-image :width="500" src="src/assets/normalization-params.png" />
          <h3>
            其中，如果需要使用到保存的训练数据的参数对样本数据进行无量纲化处理，则应通过读取'model-filepath'路径指向的pickle文件获取训练数据参数的记录
          </h3>
          <!-- <h3>在获取到原数据的存放路径后，通过numpy中的load()读取原数据，即可使用相应的插值算法对其进行插值处理。</h3> -->
          <h3>此外，在加载数据的时候，也需要根据处理的数据的类型分类讨论：</h3>
          <a-image :width="500" src="src/assets/normalization-input.png"></a-image>
          <h3>
            如上代码示例中，如果处理的数据为提取的信号特征样本，则应该将输入样本作为.pkl类型的文件进行读取，读取到的数据类型为pandas中的DataFrame；
          </h3>
          <h3>
            而如果处理的数据为信号序列，则应该将输入样本作为.npy类型的文件进行读取，读取到的数据类型为numpy数组
          </h3>
          <h2>2. 无量纲处理以及输出结果</h2>
          <h3>在进行无量纲处理时，根据处理的数据的类型不同，处理方式也不同：</h3>
          <a-image :width="550" src="src/assets/normalization-process-output.png" />
          <h3>
            (1)如果是对于提取的特征样本，则应该在步骤1中读取保存了训练数据的记录之后，再根据该记录的训练数据的参数对输入的样本进行无量纲化，以scikit-learn中的无量纲化方法为例，
            通过已加载的参数记录对象的'transform'方法对特征样本进行标准化。同时，应该注意要提前选定训练数据使用到的特征，如上所示代码中的'choose_features'。
          </h3>
          <h3>
            (2)如果是对于信号序列，以scikit-learn中的''无量纲化方法为例，将输入数据转换为合适的形状后即可对其进行无量纲化处理。需要注意的是，主程序输入的信号序列为(1,
            2048)的numpy数组
          </h3>
          <h2>3. 专有算法代码模板示例</h2>
          <h3>无量纲化的专有算法模板代码如下：</h3>
          <a-image :width="500" src="src/assets/normalization-outline-1.png"></a-image>
          <a-image :width="500" src="src/assets/normalization-outline-2.png"></a-image>
        </div>
      </div>
      <!-- 健康评估的专有算法模板 -->
      <div v-if="templateName === '健康评估'">
        <div>
          <h1>基本结构</h1>
          <h2>1. 数据输入</h2>
          <h3>健康评估的专有算法作为脚本运行时，需要从主程序获取参数：</h3>
          <a-image :width="550" src="src/assets/health-evaluation-params-1.png"></a-image>
          <h3>其中较为重要的参数为：</h3>
          <h3>(1) --model-filepath， 健康评估算法所使用评估模型的模型参数的存放路径</h3>
          <h3>(2) --input-filepath， 样本数据的存放路径</h3>
          <h3>(3) --save-filepath，处理结果的存放路径</h3>
          <h3>
            其余参数为层级评估模型所使用的各个参数，目前不建议修改，推荐使用默认参数。
          </h3>
          <h3>数据输入方法：通过python中的argparse库进行命令行参数的添加和解析</h3>
          <a-image :width="500" src="src/assets/health-evaluation-params-2.png" />

          <h2>2. 模型加载及模型推理</h2>

          <h3>
            层级评估模型的参数应该保存为.pkl的形式，并以.pkl文件的形式加载评估模型参数
          </h3>
          <a-image :width="500" src="src/assets/health-evaluation-load-model.png" />

          <h3>
            通过加载的评估模型，计算出权重矩阵，之后计算出层级权重指标，将计算得出的层级权重指标保存为.npy文件。
          </h3>
          <a-image :width="500" src="src/assets/health-evaluation-prediction.png" />
          <h2>3. 输出结果</h2>
          <h3>
            通过上述的模型推理得到层级权重指标矩阵之后，需要根据状态隶属度绘制出健康评估的可视化结果并给出相关建议。这里注意需要绘制层级指标权重柱状图，
          </h3>
          <h3>层级指标的树状图，并保存到'save_filepath'指出的路径下</h3>
          <a-image :width="500" src="src/assets/health-evaluation-output.png" />
          <!-- <code>
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          </code> -->

          <h2>4. 私有算法代码模板示例</h2>
          <h3>健康评估的专有算法模板代码如下：</h3>
          <a
            href="src/assets/exampleCode/My-HE-Algorithm-1.py"
            download="example-health-evaluation.py"
            >点击下载示例代码源文件</a
          >
          <a-image :width="500" src="src/assets/health-evaluation-outline-1.png" />
          <a-image :width="500" src="src/assets/health-evaluation-outline-2.png" />
          <a-image :width="500" src="src/assets/health-evaluation-outline-3.png" />
        </div>
      </div>
      <!-- 小波变换的专有算法模板 -->
      <div v-if="templateName === '小波变换'">
        <div>
          <h1>基本结构</h1>
          <h2>1. 数据输入</h2>
          <h3>小波变换的专有算法作为脚本运行时，主要需要从主程序获取两个参数：</h3>
          <h3>(1) --input-filepath， 输入的样本数据的存放路径</h3>
          <h3>(2) --output-filepath， 处理结果数据的存放路径</h3>

          <h3>数据输入方法：通过python中的argparse库进行命令行参数的添加和解析</h3>
          <a-image :width="500" src="src/assets/wavelet-transform-params.png" />

          <h2>2. 数据处理及结果返回</h2>
          <h3>
            在通过'input-filepath'读取输入数据后，就可以根据定义的数据处理方法处理输入数据。
            然后将处理结果保存为.npy的文件以返回给主程序
          </h3>
          <a-image :width="500" src="src/assets/wavelet-transform-output.png" />

          <!-- <code>
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          </code> -->

          <h2>3. 私有算法代码模板示例</h2>
          <h3>小波变换的专有算法模板代码如下：</h3>
          <a
            href="src/assets/exampleCode/My-Wavelet-1.py"
            download="example-wavelet-transform.py"
            >点击下载示例代码源文件</a
          >
          <a-image :width="500" src="src/assets/wavelet-transform-outline.png"></a-image>
          <h3>其中需要注意，定义的小波变换处理方法要在同一个源文件中。</h3>
        </div>
      </div>
    </el-scrollbar>
  </a-modal>
</template>

<script setup lang="ts">
import { UploadOutlined, QuestionOutlined } from "@ant-design/icons-vue";
import { message } from "ant-design-vue";
import type { UploadProps } from "ant-design-vue";
import { ref, h, watch } from "vue";
import { Action, ElMessageBox } from "element-plus";
import { useRouter } from "vue-router";
import api from "../utils/api.js";

const canSelectPythonFile = ref(true);

// 私有算法
const privateAlgorithms = [
  "插值处理",
  "特征提取",
  "特征选择",
  "小波变换",
  "无量纲化",
  "故障诊断",
  "故障预测",
  "健康评估",
];
const templateName = ref("");

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
};

const router = useRouter();
const pythonFileList = ref<UploadProps["fileList"]>([]); //算法源文件列表
const modelFileList = ref<UploadProps["fileList"]>([]); //模型源文件列表
const uploading = ref<boolean>(false);
const form = ref({
  algorithmType: null, // 私有算法类型
  fileList: [],
  faultDiagnosisType: "machineLearning", // 故障诊断算法类型，可选值为machineLearning和deepLearning
  useLog: false, // 是否使用训练模型时的标准化方法，为true时，使用训练模型时的标准化方法，为false时，使用当前数据集的标准化方法
});

// 使用watch监测form.algorithmType的变化，当用户选择算法类型后，才可以继续上传文件
// watch(form.algorithmType, (newValue) => {
//   // 当用户选择算法类型后，才可以继续上传文件
//   if (newValue === null) {
//     canSelectPythonFile.value = true;
//   }else{
//     canSelectPythonFile.value = false;
//   }
// });

// const options = ref([
//   { value: "插值处理", label: "插值处理" },
//   { value: "特征提取", label: "特征提取" },
//   { value: "无量纲化", label: "无量纲化" },
//   { value: "特征选择", label: "特征选择" },
//   { value: "小波变换", label: "小波变换" },
//   { value: "故障诊断", label: "故障诊断" },
//   { value: "故障预测", label: "故障预测" },
//   { value: "健康评估", label: "健康评估" },
// ]);
const dialogVisible = ref(false);
const uploadPrivateAlgorithmFiles = () => {
  dialogVisible.value = true;
};

const removePythonFile: UploadProps["onRemove"] = (file) => {
  // 在删除文件列表中文件的同时，重新计算ruleOfDFA，以保证用户上传私有故障诊断算法时，同时包含用于故障诊断的模型以及模型参数文件。
  const isFaultDiagnosis = form.value.algorithmType === "故障诊断";
  const isFaultPrediction = form.value.algorithmType === "故障预测";
  const isNormalization = form.value.algorithmType === "无量纲化";
  const isHealthEvaluation = form.value.algorithmType === "健康评估";
  const isPyFile = file.type === "application/x-python-code" || file.name.endsWith(".py");
  const isPklFile = file.name.endsWith(".pkl");
  const isPthFile = file.name.endsWith(".pth");
  // 当用户选择上传故障诊断或是故障预测算法时，如果删除了python文件，则ruleOfDFA减1；如果删除了模型参数文件，则ruleOfDFA加1
  if (isFaultDiagnosis || isFaultPrediction) {
    if (isPyFile && ruleOfFDA > -1) ruleOfFDA -= 1;
    if ((isPklFile || isPthFile) && ruleOfFDA < 1) ruleOfFDA += 1;
  }
  // 当用户选择上传无量纲化算法时，如果删除了python文件，则ruleOfNor减1；如果删除了模型参数文件，则ruleOfNor加1
  if (isNormalization) {
    if (isPyFile && ruleOfNor > -1) ruleOfNor -= 1;
    if ((isPklFile || isPthFile) && ruleOfNor < 1) ruleOfNor += 1;
  }
  // 当用户选择上传健康评估算法时，如果删除了python文件，则ruleOfHev减1；如果删除了模型参数文件，则ruleOfHev加1
  if (isHealthEvaluation) {
    if (isPyFile && ruleOfHev > -1) ruleOfHev -= 1;
    if ((isPklFile || isPthFile) && ruleOfHev < 1) ruleOfHev += 1;
  }
  // 删除文件列表中用户上传的文件
  if (pythonFileList.value) {
    const index = pythonFileList.value.indexOf(file);
    const newFileList = pythonFileList.value.slice();
    newFileList.splice(index, 1);
    pythonFileList.value = newFileList;
  } else {
    console.log("fileList is undefined");
  }
};

const algorithmTypeChange = (value: string, label: any, extra: any) => {
  // 当用户选择算法类型时，清空文件列表
  console.log("value: ", value);
  selectAlgorithmType();
  canSelectPythonFile.value = false;
  // 当用户选择算法类型时，判断是否可以上传模型文件
  if (
    value === "故障诊断" ||
    value === "故障预测" ||
    value === "健康评估" ||
    value === "无量纲化"
  ) {
    if (value === "无量纲化") {
      if (!form.value.useLog) {
        canUploadModelFile.value = false;
      }
      return;
    }
    canUploadModelFile.value = true;
  } else {
    canUploadModelFile.value = false;
  }
};

// 当用户选择算法类型时，清空文件列表
const selectAlgorithmType = () => {
  if (pythonFileList.value) {
    pythonFileList.value.forEach((file) => {
      removePythonFile(file);
    });
  }
};

// 算法类型为树形选择器
// const algorithmType = ref();
const treeData = ref([
  {
    label: "预处理组件",
    value: "预处理组件",
    selectable: false,
    children: [
      {
        label: "插值处理",
        value: "插值处理",
      },
      {
        label: "特征提取",
        value: "特征提取",
      },
      {
        label: "小波变换",
        value: "小波变换",
      },
      {
        label: "特征选择",
        value: "特征选择",
      },
      {
        label: "无量纲化",
        value: "无量纲化",
      },
    ],
  },
  {
    label: "故障检测组件",
    value: "故障检测组件",
    selectable: false,
    children: [
      {
        label: "故障诊断",
        value: "故障诊断",
      },
      {
        label: "故障预测",
        value: "故障预测",
      },
    ],
  },
  {
    label: "健康评估组件",
    value: "健康评估组件",
    selectable: false,
    children: [
      {
        label: "健康评估",
        value: "健康评估",
      },
    ],
  },
]);

//根据用户选择上传的私有算法的类型设置文件上传数量
const fileCount = ref(1);

let ruleOfFDA = 0;
let ruleOfNor = 0;
let ruleOfHev = 0;

const beforeUploadAlgorithmFile = (file: any) => {
  // 判断传入的文件类型是否为python文件，限定用户只能上传python文件
  const isPyFile = file.type === "application/x-python-code" || file.name.endsWith(".py");
  if (!isPyFile) {
    message.warning("请上传包含该算法的python文件");
    return;
  } else {
    let fileNum = pythonFileList.value?.length;
    if (fileNum == 1) {
      if (isPyFile) {
        removePythonFile(file);
        pythonFileList.value = [...(pythonFileList.value || []), file]; //将文件添加到fileList中
        message.warning("最多只能上传一个.py类型的文件");
        return false;
      }
      message.warn("上传该类型算法时，最多只能上传一个.py类型的文件");
      return false;
    }
    // 将文件加入到文件列表
    pythonFileList.value = [...(pythonFileList.value || []), file]; //将文件添加到fileList中
  }
};

// 删除模型文件
const removeModelFile: UploadProps["onRemove"] = (file) => {
  // 删除文件列表中用户上传的文件
  if (modelFileList.value) {
    const index = modelFileList.value.indexOf(file);
    const newFileList = modelFileList.value.slice();
    newFileList.splice(index, 1);
    modelFileList.value = newFileList;
  } else {
    console.log("fileList is undefined");
  }
};

// 上传模型文件
const canUploadModelFile = ref(false);
const beforeUploadModelFile = (file: any) => {
  let isFaultDiagnosis = form.value.algorithmType === "故障诊断" ? true : false;
  let isHealthEvaluation = form.value.algorithmType === "健康评估" ? true : false;
  let isNormalization = form.value.algorithmType === "无量纲化" ? true : false;
  let uploadModelFileType;
  let isPklFile = file.name.endsWith(".pkl");
  let isPthFile = file.name.endsWith(".pth");
  let faultDiagnosisType;
  if (isFaultDiagnosis) {
    // 如果上传故障诊断组件，如果是机器学习的故障诊断，需要上传.pkl的文件，深度学习的故障诊断需要上传.pth的文件
    faultDiagnosisType = form.value.faultDiagnosisType;
    if (faultDiagnosisType === "machineLearning") {
      if (!isPklFile) {
        message.warning("上传基于机器学习的算法，请上传.pkl的模型文件");
        return;
      }
      uploadModelFileType = "pkl";
    } else {
      if (!isPthFile) {
        message.warning("上传基于深度学习的算法，请上传.pth的模型文件");
        return;
      }
      uploadModelFileType = "pth";
    }
  }

  if (isHealthEvaluation) {
    if (!isPklFile) {
      message.warning("上传健康评估算法，请上传.pkl的模型文件");
      return;
    }
  }

  if (isNormalization) {
    if (form.value.useLog) {
      if (!isPklFile) {
        message.warning("上传对于所提取特征的无量纲化算法，请上传.pkl的模型文件");
        return;
      }
    }
  }

  let fileNum = modelFileList.value?.length;
  if (fileNum == 1) {
    if (uploadModelFileType === "pkl") {
      if (isPklFile) {
        removeModelFile(file);
        modelFileList.value = [...(modelFileList.value || []), file]; //将文件添加到fileList中
        message.warning("最多只能上传一个.pkl类型的模型文件");
        return
      }
      message.warn("上传该类型算法时，最多只能上传一个.pkl类型的文件");
      return
    }else{
      if (isPthFile){
        removeModelFile(file);
        modelFileList.value = [...(modelFileList.value || []), file]; //将文件添加到fileList中
        message.warning("最多只能上传一个.pth类型的模型文件");
        return
      }
      message.warn("上传该类型算法时，最多只能上传一个.pth类型的文件");
      return
    }
  }

  // 将文件添加到modelFileList中
  modelFileList.value = [...(modelFileList.value || []), file];
};
// const beforeUpload = (file: any) => {
//   const algorithmType = form.value.algorithmType;
//   const isFaultDiagnosis = algorithmType === "故障诊断";
//   const isFaultPrediction = algorithmType === "故障预测";
//   const isNormalization = algorithmType === "无量纲化";
//   const isHealthEvaluation = algorithmType === "健康评估";
//   const isPyFile = file.type === "application/x-python-code" || file.name.endsWith(".py");
//   const isPklFile = file.name.endsWith(".pkl");
//   const isPthFile = file.name.endsWith(".pth");

//   let fileNum = pythonFileList.value?.length;
//   if (fileNum == 0) {
//     ruleOfFDA = 0;
//     ruleOfNor = 0;
//     ruleOfHev = 0;
//   }
//   // 对于上传私有故障诊断、故障预测算法、健康评估算法的规则判断
//   if (isFaultDiagnosis || isFaultPrediction || isHealthEvaluation) {
//     // 判断用户上传的私有算法文件是否是python文件
//     if (fileNum == 0 && !isPyFile && !isPklFile && !isPthFile) {
//       message.error(
//         "请上传包含该" +
//           algorithmType +
//           "算法(.py类型)或是模型参数(.pth或是.pkl类型)的文件"
//       );
//       return false;
//     }
//     //判断是否是用户上传私有算法时是否同时上传了故障诊断的模型以及相关的模型参数文件（共两个文件）
//     if (fileNum == 1) {
//       if (
//         !(isPyFile && ruleOfFDA == -1) &&
//         !((isPthFile || isPklFile) && ruleOfFDA == 1) &&
//         !(isPyFile && ruleOfHev == -1) &&
//         !(isPklFile && ruleOfHev == 1)
//       ) {
//         message.error(
//           "上传" +
//             algorithmType +
//             "算法时，需要上传定义的" +
//             algorithmType +
//             "算法(.py文件)以及诊断时使用的模型的加载参数(.pkl或.pth文件)"
//         );
//         return false;
//       }
//     }

//     if (fileNum == 2) {
//       message.error(
//         "上传" + algorithmType + "算法时，最多只能上传包含该算法的源文件和模型参数文件"
//       );
//       return false;
//     }
//   } else if (isNormalization) {
//     // 对于上传无量纲化算法的规则判断
//     const useLog = form.value.useLog; // 为true时，对原始信号进行标准化，为false时，对提取到的特征进行标准化

//     if (!useLog) {
//       // 对原始信号进行无量纲化
//       if (fileNum == 0 && !isPyFile) {
//         message.error("请上传包含该算法的.py类型的文件");
//         return false;
//       } else {
//         // 当选择上传无量纲化私有算法时，文件列表中最多只能有一个py文件
//         if (fileNum == 1) {
//           if (isPyFile) {
//             handleRemove(file);
//           } else {
//             message.error("上传无量纲化算法时，最多只能上传一个包含该算法的python源文件");
//             return false;
//           }
//         }
//       }
//     } else {
//       // 对提取到的特征进行无量纲化
//       // 判断用户上传的私有算法文件是否是python文件
//       if (fileNum == 0 && !isPyFile && !isPklFile) {
//         message.error(
//           "请上传包含该" + algorithmType + "算法(.py类型)或是模型参数(pkl类型)的文件"
//         );
//         return false;
//       }
//       //判断是否是用户上传保存了训练数据参数的无量纲化算法时是否同时上传了无量纲化算法以及相关的模型文件（共两个文件）
//       if (fileNum == 1) {
//         if (!(isPyFile && ruleOfNor == -1) && !(isPklFile && ruleOfNor == 1)) {
//           message.error(
//             "上传" +
//               algorithmType +
//               "算法时，需要上传定义的" +
//               algorithmType +
//               "算法(.py文件)以及诊断时使用的模型的加载参数(.pkl文件)"
//           );
//           return false;
//         }
//       }
//       if (fileNum == 2) {
//         message.error(
//           "上传" + algorithmType + "算法时，最多只能上传包含该算法的源文件和模型参数文件"
//         );
//         return false;
//       }
//     }
//   } else {
//     if (fileNum == 0 && !isPyFile) {
//       message.error("请上传包含该算法的.py类型的文件");
//       return false;
//     }
//     // 当选择上传除故障诊断、故障预测、健康评估算法的其他专有算法时，文件列表中最多只能有一个py文件
//     if (fileNum == 1) {
//       if (isPyFile) {
//         handleRemove(file);
//         pythonFileList.value = [...(pythonFileList.value || []), file]; //将文件添加到fileList中
//         return false;
//       }
//       message.warn("上传该类型算法时，最多只能上传一个.py类型的文件");
//       return false;
//     }
//   }

//   pythonFileList.value = [...(pythonFileList.value || []), file]; //将文件添加到fileList中
//   // 用于判断是否是用户上传的私有故障诊断算法时是否同时上传了故障诊断的模型以及相关的模型参数文件
//   if ((isFaultDiagnosis || isFaultPrediction) && isPyFile) {
//     ruleOfFDA += 1;
//   }
//   if ((isFaultDiagnosis || isFaultPrediction) && (isPklFile || isPthFile)) {
//     ruleOfFDA -= 1;
//   }

//   // 用于判断用户上传的模型训练时的无量纲化算法是否同时上传了无量纲算法以及相关的模型文件
//   if (isNormalization) {
//     if (isPyFile) ruleOfNor += 1;
//     if (isPklFile || isPthFile) ruleOfNor -= 1;
//   }

//   // 用于判断用户上传的健康评估算法是否同时上传了健康评估算法以及相关的模型文件
//   if (isHealthEvaluation) {
//     if (isPyFile) ruleOfHev += 1;
//     if (isPklFile) ruleOfHev -= 1;
//   }
//   // if (isNormalization && (isPklFile || isPthFile)){
//   //   ruleOfNor -= 1;
//   // }
//   // console.log('beforeUpload: ', ruleOfFDA)
//   // console.log('beforeUpload: ', ruleOfFDA)
//   return false; // 阻止默认上传行为
// };

// const uploadExtraModule = () => {
//   if (!form.value.algorithmType) {
//     message.error("请选择算法类型");
//     return;
//   }
//   let isFaultDiagnosis = form.value.algorithmType === "故障诊断" ? true : false;
//   const formData = new FormData();
//   // fileList.value.forEach((file: UploadProps['fileList'][number]) => {
//   //   formData.append('files[]', file as any);
//   // });
//   let fileNum = pythonFileList.value?.length;

//   if (fileNum == 0) {
//     message.error("请选择上传包含算法的源文件");
//     return;
//   }

//   if (fileNum == 1) {
//     let algorithmType = form.value.algorithmType;
//     if (
//       algorithmType != "故障诊断" &&
//       algorithmType != "故障预测" &&
//       !(algorithmType == "无量纲化" && form.value.useLog) &&
//       algorithmType != "健康评估"
//     ) {
//       //上传除了故障诊断、故障预测、健康评估、保存训练数据参数的无量纲化以外的算法时，只能上传一个包含该算法的python源文件
//       if (pythonFileList.value) {
//         let isPyFile = pythonFileList.value[0].name.endsWith(".py");
//         if (!isPyFile) {
//           message.error("请选择上传包含该算法的py源文件");
//           return;
//         }
//         let file = pythonFileList.value[0];
//         formData.append("algorithmFile", file);
//       }
//     } else {
//       //上传专有故障诊断、专有故障预测算法、专有健康评估算法、记录训练数据参数的专有无量纲化算法时，需要同时包含定义算法文件与模型参数文件
//       if (pythonFileList.value) {
//         let isPyFile = pythonFileList.value[0].name.endsWith(".py");
//         let isPklFile = pythonFileList.value[0].name.endsWith(".pkl");
//         let isPthFile = pythonFileList.value[0].name.endsWith(".pth");
//         if (isPyFile) {
//           if (algorithmType == "无量纲化" && form.value.useLog) {
//             message.error(
//               "如果上传保存了训练数据参数的无量纲化方法，则需要同时上传模型文件(.pkl类型)"
//             );
//             return;
//           }

//           if (isFaultDiagnosis) {
//             let isMachineLearning =
//               form.value.faultDiagnosisType === "machineLearning" ? true : false;
//             if (isMachineLearning) {
//               message.error(
//                 '上传基于机器学习的故障诊断算法时，需上传模型".pkl类型"的模型文件'
//               );
//             } else {
//               message.error(
//                 '上传基于深度学习的故障诊断算法时，需上传模型".pth类型"的模型文件'
//               );
//             }
//             return;
//           }
//         }

//         if (isPklFile || isPthFile) {
//           message.error(
//             "上传" +
//               algorithmType +
//               "算法时，需上传定义的" +
//               algorithmType +
//               "算法(.py类型)"
//           );
//           return;
//         }
//       }
//     }
//   } else {
//     if (!isFaultDiagnosis) {
//       message.error("上传非故障诊断的算法时，仅需要上传一个包含该算法的py源文件");
//       return;
//     }
//     //上传私有故障诊断算法
//     let algorithm;
//     let modelParams;
//     if (pythonFileList.value && fileNum) {
//       for (let i = 0; i < fileNum; i++) {
//         if (pythonFileList.value[i].name.endsWith(".py")) {
//           formData.append("algorithmFile", pythonFileList.value[i]);
//           algorithm = pythonFileList.value[i].name.split(".")[0];
//         } else {
//           formData.append("modelParamsFile", pythonFileList.value[i]);
//           modelParams = pythonFileList.value[i].name.split(".")[0];
//         }
//       }
//       if (algorithm !== modelParams) {
//         let algorithmType = form.value.algorithmType;

//         message.error(
//           "上传定义的" +
//             algorithmType +
//             "算法(.py文件)以及使用的模型的加载参数(.pkl或.pth文件)时，两个文件名称需要保持一致"
//         );
//         return;
//       }
//     }
//   }

//   formData.append("algorithm_type", form.value.algorithmType);
//   formData.append("faultDiagnosisType", form.value.faultDiagnosisType);

//   uploading.value = true;

//   api
//     .post("/user/upload_user_private_algorithm/", formData)
//     .then((response: any) => {
//       if (response.data.code == 200) {
//         pythonFileList.value = [];
//         uploading.value = false;
//         message.success("算法文件上传成功");
//         dialogVisible.value = true;
//         ruleOfFDA = 0;
//       } else {
//         uploading.value = false;
//         message.error("算法文件上传失败, " + response.data.message);
//       }
//       if (response.data.code == 401) {
//         ElMessageBox.alert("登录状态已失效，请重新登陆", "提示", {
//           confirmButtonText: "确定",
//           callback: (action: Action) => {
//             router.push("/");
//           },
//         });
//       }
//     })
//     .catch((error: any) => {
//       uploading.value = false;
//       message.error("上传失败, 请重试");
//     });
// };

const uploadExtraModule = () => {
  let isFaultDiagnosis = form.value.algorithmType == "故障诊断" ? true : false;
  let isHealthEvaluation = form.value.algorithmType == "健康评估" ? true : false;
  let isFaultPrediction = form.value.algorithmType == "故障预测" ? true : false;
  let scalerForFeatures = (form.value.algorithmType == "无量纲化" && form.value.useLog) ? true : false;

  let pythonFileName = pythonFileList.value[0].name.split('.')[0];
  if ( isFaultDiagnosis || isHealthEvaluation || isFaultPrediction || scalerForFeatures ){
    let modelFileName = modelFileList.value[0].name.split('.')[0];
    if (pythonFileName !== modelFileName) {
      let algorithmType = form.value.algorithmType;
      message.error(
        "上传定义的" +
          algorithmType +
          "算法(.py文件)以及使用的模型的加载参数(.pkl或.pth文件)时，两个文件名称需要保持一致"
      );
      return;
    }
  }

  let formData = new FormData();
  // 发送文件上传请求
  formData.append("algorithm_type", form.value.algorithmType);
  formData.append("faultDiagnosisType", form.value.faultDiagnosisType);

  // 将pythonFileList和modelFileList中的文件添加到formData中
  for (let i = 0; i < pythonFileList.value.length; i++) {
    formData.append("algorithmFile", pythonFileList.value[i]);
  }
  for (let i = 0; i < modelFileList.value.length; i++) {
    formData.append("modelParamsFile", modelFileList.value[i]);
  }

  uploading.value = true;

  api
    .post("/user/upload_user_private_algorithm/", formData)
    .then((response: any) => {
      if (response.data.code == 200) {
        pythonFileList.value = [];
        modelFileList.value = [];
        uploading.value = false;
        message.success("算法文件上传成功");
        dialogVisible.value = true;
        ruleOfFDA = 0;
      } else {
        uploading.value = false;
        message.error("算法文件上传失败, " + response.data.message);
      }
      if (response.data.code == 401) {
        ElMessageBox.alert("登录状态已失效，请重新登陆", "提示", {
          confirmButtonText: "确定",
          callback: (action: Action) => {
            router.push("/");
          },
        });
      }
    })
    .catch((error: any) => {
      uploading.value = false;
      message.error("上传失败, 请重试");
    })
}

const showDrawer = ref(false);

const onClose = () => {
  showDrawer.value = false;
};
</script>

<style scoped>
.upload-button {
  width: 180px;
}

.private-algorithm-button {
  background-color: #ffd541;
  color: #566f4f;
  font-size: 17px;
  font-weight: 600;
  border: 1px solid #789b6e;
}
</style>
