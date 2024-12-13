<template>
  <!--
  更新页面 --v5.7
  版本  2024-10-19
  -->
  <div style="height: 100vh; overflow:hidden" @mouseover="background_IMG">
    <el-container class="fullscreen_container">
      <!-- 头部栏目 -->
      <el-header class="header">
        <!-- 左侧的 Logo 区域 -->
        <div class="logo">
          <img src="../assets/system-logo.png" alt="Logo"/>
        </div>

        <!-- 中间的标题 -->
        <div class="title">
          <h1>轨道车辆智能运维通用算法和工具软件</h1>
        </div>

        <!-- 右侧的用户操作区 -->
        <div class="user-actions">
          <!-- 欢迎信息 -->
          <span class="welcome-message">{{
              username === '用户名未设置' ? '用户名未设置' : ('欢迎' + username + '！')
            }}</span>

          <!-- 帮助 -->
          <div class="clickable action-item">
            <el-dropdown :trigger="['click']" class="clickable" placement="bottom-end">
              <div style="display: flex; align-items: center; justify-content: center;">
                <el-icon style="margin: 5px;" class="action-icon">
                  <QuestionFilled/>
                </el-icon>
                <a @click.prevent>
                  <span class="action-text">帮助</span>
                </a>
              </div>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item @click="operationHelpDialogVisible=true">操作指南</el-dropdown-item>
                  <el-dropdown-item @click="userHelpDialogVisible=true">使用教程</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>

          <!-- 退出 -->
          <div @click="logout" class="clickable action-item">
            <el-icon style="margin: 5px;" class="action-icon">
              <SwitchButton/>
            </el-icon>
            <span class="action-text">退出</span>
          </div>

          <!-- 全屏 -->
          <div @click="toggleFullscreen()" class="clickable action-item" style="margin-right: 0;">
            <el-icon style="margin: 5px;" class="action-icon">
              <FullScreen/>
            </el-icon>
            <span class="action-text">全屏</span>
          </div>
        </div>

        <!-- 操作指南 -->
        <el-dialog v-model="operationHelpDialogVisible" title="操作指南" width="810" draggable
                   :close-on-click-modal="false" :center="false">
          <div style="text-align: left;">
            <el-scrollbar height="500px">
              <h1>1、选择算法</h1>
              <h3>从算法选择区中将所选择算法模块拖拽至可视化建模区并调整位置</h3>
              <img src="../assets/step_1_drag.gif" alt="" style="width: 700px; height: auto">
              <h1>2、调整参数</h1>
              <h3>鼠标右击可视化建模区中的算法模块，在弹出的下拉框中更改算法的参数</h3>
              <img src="../assets/step_2_setting.gif" alt="" style="width: 700px; height: auto">
              <h1>3、建立流程</h1>
              <h3>鼠标移至作为所建立流程的起点的算法模块上的红色附着点，点击并拖拽至所建立流程的目标算法模块</h3>
              <img src="../assets/step_3_modeling.gif" alt="" style="width: 700px; height: auto">
              <h1>4、检查模型及修改模型</h1>
              <h3>在确保所有模块都已经建立流程后，点击完成建模，然后点击检查模型对所建立的模型进行检查，</h3>
              <h3>如果模型中存在错误，点击修改模型并根据提示对模型进行修改，然后依次点击完成建模和模型检查，</h3>
              <h3>通过模型检查后即可以保存模型，并进行后续操作。</h3>
              <img src="../assets/step_4_modelCheck.gif" alt="" style="width: 750px; height: auto">
            </el-scrollbar>
          </div>
        </el-dialog>

        <!-- 使用教程 -->
        <el-dialog v-model="userHelpDialogVisible" title="使用教程" width="810" draggable :close-on-click-modal="false"
                   :center="false">
          <div style="text-align: left;">
            <el-scrollbar height="500px" ref="userHelpDialogScrollbar">
              <h2>常见问题</h2>

              <div id="howToUseThisApp">
                <h2>1、如何使用本软件？</h2>
                <h3>本软件分为五个部分：①算法选择区、②数据加载区、③可视化建模区、④结果可视化区、⑤加载模型区。</h3>
                <img src="../assets/system-outline.png" alt="" style="width: 700px; height: auto">
                <h3>①<span style="color: red;">数据加载区</span>用于上传数据文件，查看用户的历史文件以及加载数据文件。
                </h3>
                <h3>②<span style="color: red;">算法选择区</span>中包含本系统支持的算法模块，可以选择其中的算法模块并拖入到可视化建模区进行建模。
                </h3>
                <h3>③<span style="color: red;">可视化建模区</span>是进行建模的区域，其中包含所有已经建立流程的算法模块，可以拖拽模块进行位置调整，也可以右击模块进行参数调整。
                </h3>
                <h3>④<span style="color: red;">结果可视化区</span>用于查看模型运行结果，包括模型运行结果图、模型运行结果表格。
                </h3>
                <h3>⑤<span style="color: red;">加载模型区</span>用于查看用户曾经保存的历史模型，和加载已保存的模型。</h3>
              </div>

              <div>
                <h2>2、如何在算法选择区中选择对应算法？</h2>
                <h3>在算法选择区中，每个模块都有三级展开结构，</h3>
                <img src="../assets/algorithms-unfold.png" alt="" style="width: 300px; height: auto">
                <h3>点击对应模块展开到最后一级，这时再点击对应算法会在结果可视化区中呈现该算法的介绍。</h3>
                <img src="../assets/algorithm-introduction.png" alt="" style="width: 700px; height: auto">
                <h3>通过拖动的方式可以将指定的算法模块拖入可视化建模区。并可以通过点击右键调整算法参数。</h3>
                <img src="../assets/set-params.png" alt="" style="width: 700px; height: auto">

              </div>

              <div>
                <h2>3、如何建立模型并保存？</h2>
                <h3>
                  在算法选择区中选择对应算法拖入到可视化建模区后，通过点击可视化建模区中算法模块上右侧的红色附着点，可以拉取一条连线并可连接到另一个算法模块上，以此来表示模型的运行顺序。</h3>
                <img src="../assets/line.gif" alt="" style="width: 700px; height: auto">
                <h3>在建立好模型后，点击完成建模，此时可以点击检查模型进行模型检查，具体操作步骤如下。</h3>
                <h3>第一步，点击完成建模</h3>
                <img src="../assets/modeling-finish-1.png" alt="" style="width: 700px; height: auto">
                <h3>第二步，点击检查模型。如果模型中存在错误，会呈现错误提示。</h3>
                <img src="../assets/modeling-finish-2.png" alt="" style="width: 700px; height: auto">
                <img src="../assets/check-model-tip.png" alt="" style="width: 700px; height: auto">

                <h3>第三步，如果模型中存在错误，点击修改模型并根据提示对模型进行修改，</h3>
                <img src="../assets/rectify-model-1.png" alt="" style="width: 700px; height: auto">
                <h3>
                  点击修改模型，此时提示正在修改模型，标红的连线表示该处存在逻辑错误，可以点击模块右上方红色的删除按钮删除报错模块，</h3>
                <img src="../assets/rectify-model-2.png" alt="" style="width: 700px; height: auto">
                <h3>删除报错模块后，正确建立模型流程，然后重复上述过程，直到通过模型检查，</h3>
                <img src="../assets/rectify-model-3.png" alt="" style="width: 700px; height: auto">

                <h3>第四步，完成上述流程后，点击保存模型进行模型的保存，</h3>
                <img src="../assets/save-model-1.png" alt="" style="width: 700px; height: auto">
                <h3>输入模型名称，点击确定，</h3>
                <img src="../assets/save-model-2.png" alt="" style="width: 700px; height: auto">

                <h3>其中建模时的模型检查遵循如下流程图中的规则，</h3>
                <img src="../assets/modeling-processing.png" alt="" style="width: 300px; height: auto">

              </div>

              <div>
                <h2>4、如何查看历史模型？</h2>
                <h3>在加载模型区中，点击用户历史模型，</h3>
                <img src="../assets/browser-saved-models-1.png" alt="" style="width: 700px; height: auto">
                <h3>
                  此时左侧弹窗中显示的就是用户保存过的历史模型，并且可以点击使用复现历史模型，点击删除历史模型，或是点击查看历史模型信息，</h3>
                <img src="../assets/browser-saved-models-2.png" alt="" style="width: 700px; height: auto">
                </img>
              </div>

              <div>
                <h2>5、如何上传文件到服务器？</h2>
                <h3>第一步，在数据加载区中，点击本地文件，</h3>
                <img src="../assets/upload-data-1.png" alt="" style="width: 700px; height: auto">

                <h3>
                  点击选择文件，选择本地文件，使其加载到文件列表中，每次只可以上传一个文件，并且可以点击文件列表中的删除图标要上传的文件，</h3>
                <img src="../assets/upload-data-3.png" alt="" style="width: 300px; height: auto">
                <h3>第二步，点击上传至服务器，根据提示输入文件名与文件描述，点击确定进行上传</h3>
                <img src="../assets/upload-data-2.png" alt="" style="width: 700px; height: auto">
              </div>

              <!-- <a href="javascript:void(0);" @click="scrollTo('howToUseThisApp')">1、如何使用本软件？</a>  -->

            </el-scrollbar>
          </div>
        </el-dialog>
      </el-header>

      <!-- 可视化建模主界面 -->
      <el-container>

        <!-- 左侧菜单栏 -->
        <el-aside width="250"
                  style="border: rgb(95,117,154) 1px solid;margin: 3px;border-radius: 8px;box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);background: white;overflow-x: hidden;">
          <!-- #80a5ba -->
          <div class="aside-title"
               style="font-size: 25px; color: #56a4e4;border: rgb(204,208,214) 1px solid;width: 100%;">
            <img src="../assets/algorithms-icon.svg"
                 style="width: 40px; height: auto; color: #34374f"/><span>模型构建</span>
          </div>
          <el-divider style="height: 15px;background: #CDCFD0;margin-top: 0;margin-bottom: 0;"/>
          <!-- #eff3f6 -->
          <div class="algorithms-selection" style="background: white">
            <!-- 选择组件或者打开模型库 -->
            <div style="height: 90%">
              <!--      m1      -->
              <div style="border: rgb(204,208,214) 1px solid;background: rgb(204, 208, 214);padding: 1px;">
                <!-- 选择自定义的模型，或者预定义的模型 -->
                <div
                    style="height: 40px;width: 100%; background-color: rgb(204, 208, 214); border-bottom: #CDCFD0 1px solid;margin-top: 2px;">
                  <!-- 系统用户可以自由选择自定义模型或者使用历史模型 -->
                  <!-- <a-radio-button value="customModel" style="width: 50%; border: none; border-radius: 0; font-weight: bolder; font-size: large; color:#558b48">基础组件</a-radio-button>
                  <a-radio-button value="templateModel" style="width: 50%; border: none; border-radius: 0; font-weight: bolder; font-size: large">历史模型</a-radio-button> -->
                  <a-radio-group v-if="userRole === 'superuser' || userRole === 'user'" v-model:value="modelSelection"
                                 button-style="solid" size="large"
                                 style="padding: 0; width: 100%; height: 100%;">
                    <el-tooltip placement="right-start"
                                :content="'选择组件进行建模'"
                                effect="light">
                      <a-radio-button value="customModel" :style="getRadioButtonStyle('customModel')"
                                      class="custom-radio-button">
                        基础组件
                      </a-radio-button>
                    </el-tooltip>
                    <!--                  <a-radio-button value="templateModel" :style="getRadioButtonStyle('templateModel')" class="custom-radio-button">系统模型库</a-radio-button>-->
                  </a-radio-group>
                </div>
                <!-- 系统用户可以使用基础组件定义模型 -->
                <!--              <div-->
                <!--                  style="background-color: white;-->
                <!--                color: #436f41;-->
                <!--                font-family: 'Arial', sans-serif;-->
                <!--                font-size: 18px;-->
                <!--                line-height: 30px;-->
                <!--                text-align: center;-->
                <!--                border-bottom: solid 1px #CDCFD0;-->
                <!--                border-bottom-left-radius: 29px;-->
                <!--                border-bottom-right-radius: 29px;-->
                <!--                width: 100%;-->
                <!--                height: 30px;" v-if="((userRole === 'superuser' || userRole === 'user') )">选择组件进行建模-->
                <!--              </div>-->
                <!-- 算法选择区中算法展开结构，只对系统用户可见 -->
                <div v-if="((userRole === 'superuser' || userRole === 'user') )"
                     style="width: 248px; height: 330px;background-color: white; border: 1px solid #527b96; border-radius: 5px; margin-top: 2px;">
                  <el-scrollbar height="100%" :min-size="35" style="margin-left: 10px;">
                    <!-- 数据源节点 -->
                    <!--                <div style="font-size: 20px; color: #343655; font-weight: 600">数据源组件</div>-->
                    <!--                <a-tooltip title="拖拽数据源节点至可视化建模区">-->
                    <el-tooltip placement="right-start" :content="'拖拽数据源组件至可视化建模区'"
                                effect="light">
                      <div :draggable="true" @dragend="handleDragend($event, 'dataSource', dataSourceNode)" class="item"
                           @click="showIntroduction('dataSource'.replace(/_multiple/g, ''))"
                           style="background-color: #7cbbe4;
                         border-radius: 5px; align-content: center;
                         width: 150px; height: 40px;background-image: linear-gradient(#5daefd, #89cffb); color: #1e213b;
                         margin-top: 10px;
">
                        <el-text style="width: 105px; font-size: 17px; font-weight: 600; color: #34374f;"
                                 truncated>
                          {{ labelsForAlgorithms['dataSource'] + '组件' }}
                        </el-text>
                      </div>
                    </el-tooltip>
                    <el-col v-for="item in menuList2">
                      <!-- #4599be #5A87F8 -->
                      <!-- 此为一级目录，点击展开二级目录 -->
                      <el-row>
                        <el-button
                            style="width: 150px; height: 40px; margin-top: 10px; background-image: linear-gradient(#5daefd, #89cffb); color: #1e213b; "
                            icon="ArrowDown" @click="menuDetailsSecond[item.label] = !menuDetailsSecond[item.label]">
                          <el-text style="width: 105px; font-size: 17px; color: #34374f; font-weight: 600;" truncated>{{
                              item.label
                            }}
                          </el-text>
                        </el-button>
                      </el-row>
                      <div style="border-left: solid 2px #CDCFD0; margin-left: 3px; margin-top: 3px">
                        <el-col v-if="menuDetailsSecond[item.label]" v-for="option in item.options">
                          <!-- 此为二级目录，点击展开三级目录 -->
                          <el-row style="margin-left: 20px;">
                            <!--  #75acc3 -->
                            <el-button
                                style="width: 150px; margin-top: 7px; background-color: #72A1DB; border: 0px; background-image: linear-gradient(#a0d9fd, #a8d8ff);"
                                type="info" @click="clickAtSecondMenu(option)">
                            <span><img :src="setIconOfAlgorithms(option.label)" alt=""
                                       style="position:absolute; left: 5px; top: 12px; width: 20px; height: auto;"></span>
                              <el-text style="width: 105px; font-size: 15px; color: #343655; font-weight: 600;"
                                       truncated>
                                {{ option.label }}
                              </el-text>
                            </el-button>
                          </el-row>
                          <div style="margin-left: 20px;border-left: solid 2px #CDCFD0;">
                            <el-row v-if="menuDetailsThird[option.label]"
                                    v-for="algorithm in Object.keys(option.parameters)"
                                    style="display: flex; align-items: center;">
                              <span
                                  style="width: 20px; height: 30px; display: flex; align-items: center; justify-content: center; background-color: white;">
                                <div style="height: 2px; width: 100%; background-color: #CDCFD0;"></div>
                              </span>
                              <el-tooltip placement="right-start" :content="labelsForAlgorithms[algorithm]"
                                          effect="light">
                                <!-- #f9fcff -->
                                <!-- 此为三级目录，点击进行算法选择 -->
                                <div :draggable="true" @dragend="handleDragend($event, algorithm, option)" class="item"
                                     @click="showIntroduction(algorithm.replace(/_multiple/g, ''))"
                                     style="background-color: #f9eeed ; margin-top: 7px; width: 145px; height: 30px; margin-bottom: 10px; padding: 0px; border: 1px solid #3473d5; border-radius: 5px; align-content: center;">
                                  <el-text style="width: 105px; font-size: 14px; font-weight: 600; color: #3889ca;"
                                           truncated>{{
                                      labelsForAlgorithms[algorithm]
                                    }}
                                  </el-text>
                                </div>
                              </el-tooltip>
                            </el-row>
                          </div>
                        </el-col>
                      </div>
                    </el-col>

                    <!--                  针对增值服务组件动态渲染-->

                    <el-row>
                      <el-button
                          style="width: 150px; height: 40px; margin-top: 10px; background-image: linear-gradient(#5daefd, #89cffb); color: #1e213b; "
                          icon="ArrowDown" @click="isShowSecondButton=!isShowSecondButton">
                        <el-text style="width: 105px; font-size: 17px; color: #34374f; font-weight: 600;" truncated>
                          增值组件
                        </el-text>
                      </el-button>
                    </el-row>
                    <div style="border-left: solid 2px #CDCFD0; margin-left: 3px; margin-top: 3px">
                      <el-row v-if="isShowSecondButton" style="margin-left: 20px;"
                              v-for="dataSourceNode in fetchedExtraAlgorithmList">
                        <el-tooltip placement="right-start" :content="dataSourceNode.alias" effect="light">
                          <el-button :draggable="true"
                                     @dragend="handleDragendAdd($event, dataSourceNode.alias, dataSourceNode)"
                                     style="width: 150px; margin-top: 7px; background-color: #72A1DB; border: 0px; background-image: linear-gradient(#a0d9fd, #a8d8ff);">
                            <el-text style="width: 105px; font-size: 15px; color: #343655; font-weight: 600;" truncated>
                              {{ dataSourceNode.alias }}
                            </el-text>
                          </el-button>
                        </el-tooltip>
                      </el-row>
                    </div>

                  </el-scrollbar>
                  <!--                <div style="height: 100px; border-top: 1px solid #527b96; display: flex; flex-direction: column ;justify-content: center; align-items: center;">-->
                  <!--                  &lt;!&ndash; 数据源节点 &ndash;&gt;-->
                  <!--                  <div style="font-size: 20px; color: #343655; font-weight: 600">数据源组件</div>-->
                  <!--                  <a-tooltip title="拖拽数据源节点至可视化建模区" >-->
                  <!--                    <div :draggable="true" @dragend="handleDragend($event, 'dataSource', dataSourceNode)" class="item"-->
                  <!--                    @click="showIntroduction('dataSource'.replace(/_multiple/g, ''))"-->
                  <!--                    style="background-color: #7cbbe4; margin-top: 7px; width: 145px; height: 30px; margin-bottom: 10px; padding: 0px; border-radius: 5px; align-content: center;">-->
                  <!--                    <el-text style="width: 105px; font-size: 16px; font-weight: 600; color: white; font-size: 18px" truncated>-->
                  <!--                    {{ labelsForAlgorithms['dataSource'] }}</el-text>-->
                  <!--                    </div>-->
                  <!--                  </a-tooltip>-->
                  <!--                </div>-->
                </div>
                <el-divider style="height: 15px;background: #CDCFD0;margin-bottom: 0;margin-top: 0px;"/>
                <!--      m2      -->
                <div style="border: rgb(204,208,214) 1px solid;background: rgb(204, 208, 214);padding: 1px;">
                  <!-- 选择系统模型库 -->
                  <div
                      style="height: 40px; width: 100%; background-color: rgb(204, 208, 214); border-bottom: #CDCFD0 1px solid;margin-top: 2px;">
                    <!-- 系统用户可以自由选择自定义模型或者使用历史模型 -->
                    <a-radio-group v-if="userRole === 'superuser' || userRole === 'user'" v-model:value="modelSelection"
                                   button-style="solid" size="large" style=" padding: 0; width: 100%; height: 100%">
                      <!-- <a-radio-button value="customModel" style="width: 50%; border: none; border-radius: 0; font-weight: bolder; font-size: large; color:#558b48">基础组件</a-radio-button>
                      <a-radio-button value="templateModel" style="width: 50%; border: none; border-radius: 0; font-weight: bolder; font-size: large">历史模型</a-radio-button> -->

                      <el-tooltip placement="right-start" :content="'从数据库中选择模型'"
                                  effect="light">
                        <a-radio-button value="templateModel" :style="getRadioButtonStyle('templateModel')"
                                        class="custom-radio-button">系统模型库
                        </a-radio-button>
                      </el-tooltip>
                    </a-radio-group>
                  </div>
                  <div v-if="((userRole === 'superuser' || userRole === 'user') )"
                       style="position: relative; width: 248px; height: 170px; background-color: #FCFDFF; display: flex;
              justify-content: center; align-items: center; border-radius: 5px; margin-top: 2px; border: 1px solid #527b96">
                    <a-button style="width: 165px; height: 35px; font-size: 16px; position:absolute;
                top: 25px; left: 40px; display:flex; justify-content: center; align-items: center;
                background-image: linear-gradient(to bottom right, #a1a2b1, #edf4f6); color: #3c93f8;
                border: 2px solid #484a66; font-size: 18px; border-radius: 15px;
                font-weight: 550; "
                              @click="fetchModels">
                      <template #icon>
                        <!-- <EllipsisOutlined /> -->
                        <img src="../assets/systemModels.svg" alt="" width="40px" height="40px"/>
                      </template>
                      打开模型库
                    </a-button>
                    <div class="highlight" :style="{bottom: '15px', color: getColor(modelLoaded)}" :title="modelLoaded">
                      <p>
                        已加载模型</p>{{ modelLoaded }}
                    </div>

                    <!-- 供普通用户上传数据文件 -->
                    <!-- <div v-if="userRole === 'user'"
                    style="width: 100%; height: 150px; background-color: white; position: absolute; bottom: 20px; border-top: 2px solid #527b96; padding-top: 15px">
                      <div style="font-size: 20px; color: #343655; font-weight: 600; margin-bottom: 20px;">上传数据文件
                        <el-popover
                          title="上传数据格式"
                          confirm-button-text="确认"
                          trigger="hover"
                          :width="500"
                        >
                          <template #default>
                            <p>目前系统可处理的数据格式为长度为2048的信号序列，<br>
                            如果为多传感器数据则确保其数据形状为（2048，传感器数量），其中2048为信号长度，<br>
                            请按照如上的数据格式，并以.npy或是.mat的文件格式上传。</p>
                          </template>
                          <template #reference>
                            <div style="position: absolute; right: 35px; top: 15px;">
                              <a class='datatype-trigger-icon'><question-circle-outlined/></a>
                            </div>
                          </template>

                        </el-popover>
                      </div>

                      <div style="position: relative;">
                        <a-upload
                          :file-list="fileList"
                          :max-count="1"
                          @remove="handleRemove"
                          :before-upload="beforeUpload"
                        >
                          <a-button class="custom-button"
                          >
                            <template #icon>
                              <img src="../assets/folderOpen.svg" alt="" width="20">
                            </template>
                            选择本地文件
                          </a-button>
                        </a-upload>
                        <a-button

                          type="primary"
                          :disabled="fileList.length === 0"
                          :loading="uploading"
                          style="width: 165px; height: 35px;
                          font-size: 16px; border-radius: 15px;
                          "
                          @click="handleUpload"
                        >
                          <UploadOutlined />
                          {{ uploading ? "正在上传" : "上传至服务器" }}
                        </a-button>

                      </div>
                    </div> -->

                  </div>
                </div>
              </div>
              <!-- 上传增值服务组件，只对系统用户可见 -->
              <!--            <el-divider style="height: 5px;background: #CDCFD0;margin-top: 16px;margin-bottom: 0;"/>-->
              <div style="height: auto">
                <div
                    style="padding: 10px; border: 4px solid #ffd541; width: 200px;border-radius: 5px;"
                    v-if="userRole === 'superuser'">
                  <!-- <p style="padding-bottom: 5px">上传增值服务组件</p> -->
                  <div
                      style="height: 20%; width: 100%; color: #343655; font-size: 20px;
                    font-weight: 600;justify-content: center;align-items: center;margin-bottom: 10px;">
                    增值服务组件
                  </div>
                  <div style="display: flex; flex-direction: row">
                    <div style="flex: 1;margin-right: 3px;">
                      <uploadPrivateAlgorithmFiless @addExtraModule="handleAddExtraModule"/>
                    </div>
                    <div style="flex: 1;margin-left: 3px;">
                      <managePrivateAlgorithm @deleteExtraModule="handleDeleteExtraModule"/>
                    </div>
                  </div>
              </div>
            </div>
          </div>
          </div>

        </el-aside>

        <!-- 可视化建模区以及结果可视化区 -->
        <!-- <el-main @dragover.prevent ref="efContainerRef" id="efContainer "
          style="height: auto; width: 600px; padding: 0px;"> -->
        <el-main @dragover.prevent ref="efContainerRef" id="efContainer "
                 style="height: auto; width: 600px;
          border-radius: 8px;margin: 3px;padding: 2px;border: rgb(95,117,154) 1px solid;padding-bottom: 5px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);overflow-y: hidden">
          <!-- 可视化建模区的主要内容 -->
          <div
              style=" position: relative; height: 32%; font-size: 20px; color: #003e50; font-weight: 500; font-family:Arial, Helvetica, sans-serif;  background-position: center;">
            <div id="statusIndicator" class="status-indicator">未建立模型</div>

            <!-- <el-button type="primary" style="font-size: 18px; width: 180px;" @click="drawer = true">打开功能区</el-button> -->
            <!-- 可视化建模区中的算法节点 -->
            <DraggableContainer :reference-line-visible="false">
              <Vue3DraggableResizable :draggable="true" :resizable="false" v-for="(item, index) in nodeList"
                                      :key="item.nodeId" class="node-info" :id="item.id"
                                      :style="item.nodeContainerStyle"
                                      style="background-image: url('../assets/interpolation-icon.svg')"
                                      :ref="el => nodeRef[index] = el" @click="showResult(item)">
                <!-- <el-tooltip effect="dark" content="通过节点中心的附着点拖拽节点，或右击打开参数配置" :teleported="false"> -->
                <span>
                    <!-- 右键点击可视化建模区中的算法节点弹出对应的参数配置对话框 -->
                    <el-popover placement="bottom" :title="item.label + '参数配置'"
                                @show="getExtraAlgorithm(item)" :teleported="true" :width="450" trigger="contextmenu"
                                popper-style=";display: flex; flex-direction: column; align-content: left;">
                      <!-- 调整算法参数 -->
                      <!-- 可视化建模区中的各节点所具有的参数与代码中menuList2中的参数是相对应的 -->
                      <el-row
                          v-if="item.use_algorithm != null && item.id != '1.2' && item.id != '1.3' && item.id != '1.5' && (item.use_algorithm.indexOf('private')==-1) && (item.use_algorithm.indexOf('extra')==-1)"
                          v-for="(value, key) in item.parameters[item.use_algorithm]"
                          :key="item.parameters[item.use_algorithm].keys" style="margin-bottom: 20px">
                        
                        <el-col :span="8" style="align-content: center;"><span
                            style="margin-left: 10px; font-size: 15px;">{{ labelsForParams[key] }}：</span></el-col>
                        <el-col :span="16">
                          <el-select v-model="item.parameters[item.use_algorithm][key]" collapse-tags
                                     collapse-tags-tooltip :teleported="false" placeholder="请选择参数">
                            <el-option
                                v-for="item in recommendParams[key]"
                                :key="item.value"
                                :label="item.label"
                                :value="item.value"
                                style="width: 200px; height: auto; background-color: white;"
                            />
                          </el-select>
                        </el-col>
                      </el-row>
                      <div
                          v-if="item.use_algorithm != null && !item.optional && (String(item.label_display).indexOf('增值服务组件') === -1)">目前暂无可调参数</div>
                      <!-- <div v-if="item.id === '4'">对数据源的配置请在数据源区域进行</div> -->
                      <!-- 特征提取选择要显示的特征 -->
                      <el-row v-if="item.id == '1.2'">
                        <el-col :span="8" style=""><el-text style="font-size: 15px;">选择特征：<br><span
                            style="font-size: 12px;">(共{{
                            Object.keys(item.parameters[item.use_algorithm]).length
                          }}个可选特征)</span></el-text></el-col>
                        <el-col :span="16">
                          <div>
                            <el-select v-model="features" multiple collapse-tags collapse-tags-tooltip
                                       :max-collapse-tags="3" popper-class="select"
                                       placeholder="选择需要提取的特征" :teleported="false">
                              <!-- <template #header>
                                <el-checkbox
                                  v-model="checkAll"
                                  :indeterminate="indeterminate"
                                  @change="handleCheckAll"
                                >
                                  All
                                </el-checkbox>
                              </template> -->
                              <el-option v-for="(value, key) in item.parameters[item.use_algorithm]" :label="key"
                                         :value="key" style="width: 200px; background-color: white; padding: 0px;"/>
                            </el-select>
                          </div>
                        </el-col>
                      </el-row>

                      <!-- 数据源的配置项 -->
                      <div v-if="item.id == '4'">
                        <a-radio-group v-model:value="loadingDataModel">
                          <a-radio :value="1" style="margin-right: 20px">上传数据文件</a-radio>
                          <a-radio :value="2">加载数据文件</a-radio>
                        </a-radio-group>
                        <div style="background-color: white; display: flex; flex-direction: column; padding: 20px">
                          <!-- 上传数据文件和加载数据文件 -->
                          <div>
                            <div v-if="loadingDataModel === 1" style="position: relative">
                              <a-space direction="vertical">
                                <a-form :model="dataFileFormState" ref="fileUploadFormRef" @finish="onFinish">
                                  <a-form-item label="文件名称" name="filename"
                                               :rules="[{ required: true, message: '请输入文件名!' },
                                  { pattern:/^[\u4e00-\u9fa5_a-zA-Z0-9]+$/, message: '请输入中英文/数字/下划线', trigger: 'blur' }]">
                                    <a-input v-model:value="dataFileFormState.filename" placeholder="请输入文件名"/>
                                    <!-- 提示文件名命名规则 -->
                                    <div style="color: #999; font-size: 12px;">(只能包含中英文、数字、下划线)</div>
                                  </a-form-item>
                                  <a-form-item label="文件描述" name="description"
                                               :rules="[{ required: true, message: '请输入文件描述!' },{ min: 1, max: 200, message: '长度应在1到200个字符之间!', trigger: 'blur' }]">
                                    <a-input
                                        v-model:value="dataFileFormState.description"
                                        autofocus
                                        placeholder="请输入文件描述"
                                    />
                                    <!-- 提示文件描述命名规则 -->
                                    <div style="color: #999; font-size: 12px;">(长度不超过200字符)</div>
                                  </a-form-item>
                                  <a-form-item label="选择数据类型">
                                    <a-radio-group v-model:value="dataFileFormState.multipleSensors">
                                      <a-radio :value="'multiple'">多传感器数据</a-radio>
                                      <a-radio :value="'single'">单传感器数据</a-radio>
                                    </a-radio-group>
                                  </a-form-item>
                                  <a-form-item label="是否公开">
                                    <a-radio-group v-model:value="dataFileFormState.isPublic">
                                      <a-radio :value="'public'">是</a-radio>
                                      <a-radio :value="'private'">否</a-radio>
                                    </a-radio-group>
                                  </a-form-item>
                                  <a-form-item>
                                    <a-upload
                                        :file-list="fileList"
                                        :max-count="1"
                                        @remove="handleRemove"
                                        :before-upload="beforeUpload"
                                    >
                                      <a-button
                                          style="margin-top: 16px; margin-left: 0px; width: 140px; font-size: 16px; background-color: #2082F9; color: white"
                                          :icon="h(FolderOpenOutlined)">
                                        选择本地文件
                                      </a-button>

                                    </a-upload>
                                    <el-popover
                                        title="上传数据格式"
                                        confirm-button-text="确认"
                                        trigger="hover"
                                        :width="500"
                                    >
                                      <template #default>
                                        <p>目前系统可处理的数据格式为长度为2048的信号序列，<br>
                                        如果为多传感器数据则确保其数据形状为（2048，传感器数量），其中2048为信号长度，<br>
                                        请按照如上的数据格式，并以.npy或是.mat的文件格式上传。</p>
                                      </template>
                                      <template #reference>
                                        <div style="position: absolute; top: 10px; left: 150px">
                                          <a class='datatype-trigger-icon'><question-circle-outlined/></a>
                                        </div>
                                      </template>
                                    </el-popover>
                                  </a-form-item>
                                  
                                  <a-form-item>
                                    <a-button type="primary" html-type="submit" :disabled="fileList?.length === 0"
                                              :loading="uploading">
                                    <UploadOutlined/>
                                      {{ uploading ? "正在上传" : "上传至服务器" }}
                                    </a-button>
                                    
                                  </a-form-item>
                                </a-form>
                              </a-space>

                            </div>
                            <div v-if="loadingDataModel == 2">
                              <a-button
                                  type="default"
                                  style="margin-top: 25px; margin-left: 0px; width: 160px; font-size: 16px;  background-color: #2082F9; color: white"
                                  @click="switchDrawer"
                                  :icon="h(FolderOutlined)"
                              >查看已上传文件</a-button>
                            </div>


                            <!-- 确认上传文件的弹窗 -->
                            <!-- <a-modal
                              v-model:open="uploadConfirmDialog"
                              title="提交所保存文件信息"
                              :confirm-loading="uploadconfirmLoading"
                              @ok="handleOk"
                              
                              okText="确定"
                              cancelText="取消"
                              :maskClosable="false"
                              :zIndex="1000"
                              style="top: 500px"
                            >
                              <a-space direction="vertical">
                                <a-form :model="formState" :rules="fileNameRules" ref="formRef">
                                  <a-form-item label="文件名称" name="filename">
                                    <a-input v-model:value="formState.filename" placeholder="请输入文件名" />
                                  </a-form-item>
                                  <a-form-item label="文件描述" name="description">
                                    <a-input
                                      v-model:value="formState.description"
                                      autofocus
                                      placeholder="请输入文件描述"
                                    />
                                  </a-form-item>
                                </a-form>
                              </a-space>
                            </a-modal> -->
                            
                          </div>
                          <div></div>
                          <!-- 分割线 -->
                          <!-- <div style="width: 2px; height: 136px; background-color: #808080; position: absolute; right: 185px; bottom: 0px; border-radius: 1px;"></div> -->
                          <!-- 显示已加载的数据名称 -->
                          <div v-if="loadingDataModel === 2"
                               style="display: flex; position: relative;width: 100%;height: 100%">
                            <!-- <div style="font-size: 18px; font-weight: 600">已加载数据</div> -->
                            <div class="highlight"
                                 :style="{color: getColor(usingDatafile), position: 'relative', 'margin-top': '30px'}"
                                 :title="usingDatafile">已加载数据：{{ usingDatafile }}</div>
                          </div>
                          
                          
                        </div>
                        <!-- <div class="aside-title">
                          <img src="../assets/data-icon.svg" style="width: 30px; height: auto;"/><span>加载数据</span>
                        </div> -->
                        <!-- <div style="width: 250px; height: 200px; position: relative; background-color: white;">
                          <uploadDatafile @switchDrawer="handleSwitchDrawer" :api="api" />
                          <div class="highlight" :style="{color: getColor(usingDatafile)}" :title="usingDatafile">已加载数据：{{ usingDatafile }}</div>
                  
                        </div> -->
                        <!-- <a-button @click="dataSourceSetting = true">打开数据源配置</a-button> -->
                      </div>

                      <!-- 选择特征选择的规则以及设定规则的阈值 -->
                      <div v-if="item.id == '1.3' && item.use_algorithm !== 'extra_feature_selection'">
                        <el-radio-group v-model="item.parameters[item.use_algorithm]['rule']">
                          <el-radio :value="1" size="large">规则一</el-radio>
                          <el-radio :value="2" size="large">规则二</el-radio>
                        </el-radio-group>
                        <!-- 特征选择规则一 -->
                        <div v-if="item.parameters[item.use_algorithm]['rule'] == 1">
                          <div style="margin-top: 5px; margin-bottom: 15px;">
                            设定阈值后，将选择重要性系数大于该阈值的特征
                          </div>

                          <el-form>
                            <el-form-item label="阈值">
                              <!-- <el-select v-model="item.parameters[item.use_algorithm]['threshold1']" size='large' placeholder="请输入阈值" style="width: 250px;" :teleported="false">
                                <el-option 
                                v-for="item in recommendParams['threshold1'][item.use_algorithm]"
                                :key="item.value"
                                :label="item.label"
                                :value="item.value"
                                style="width: 200px; height: auto; background-color: white;" 
                                />
                              </el-select> -->
                              <el-input-number v-model="item.parameters[item.use_algorithm]['threshold1']"
                                               :precision="2" :step="0.05" :max="1" :min="0"/>
                            </el-form-item>
                          </el-form>
                        </div>
                        <!-- <template #default>
                          <div>
                            当前暂无可调参数
                          </div>
                        </template> -->
                        <!-- 特征选择规则二 -->
                        <div v-if="item.parameters[item.use_algorithm]['rule'] == 2">
                          <div style="margin-top: 5px; margin-bottom: 15px;">
                            设定阈值后，将根据特征的重要性，由高到低地选择特征，直到所选特征的重要性的总和占所有特征的重要性比例不小于该阈值，其中所有特征的重要性占比为1
                          </div>

                          <el-form>
                            <el-form-item label="阈值">
                              <!-- <el-select v-model="item.parameters[item.use_algorithm]['threshold2']" size='large' placeholder="请输入阈值" style="width: 250px;" :teleported="false">
                                <el-option 
                                v-for="item in recommendParams['threshold2'][item.use_algorithm]"
                                :key="item.value"
                                :label="item.label"
                                :value="item.value"
                                style="width: 200px; height: auto; background-color: white;" 
                                />
                              </el-select> -->
                              <el-input-number v-model="item.parameters[item.use_algorithm]['threshold2']"
                                               :precision="2" :step="0.05" :max="1" :min="0"/>
                            </el-form-item>
                          </el-form>
                        </div>
                      </div>

                      <!-- 无量纲化参数设置 -->
                      <div v-if="item.id == '1.5'">
                        <div>是否使用模型训练时的标准化方法</div>
                        <el-radio-group v-model="item.parameters[item.use_algorithm]['useLog']">
                          <el-radio :value="true" size="large">是</el-radio>
                          <el-radio :value="false" size="large">否</el-radio>
                        </el-radio-group>
                      </div>


                      <!-- <div v-if="String(item.label_display).indexOf('私有') != -1" >
                        
                        <a-select
                          v-model:value="item.parameters[item.use_algorithm]"
                          style="width: 120px"
                        >
                          <a-select-option 
                            v-for="option in privateAlgorithms" 
                            :key="option.value"
                            v-if="item.label === option.type"
                            :value="option.value" 
                            :label="option.label"
                          >{{ option.label }}</a-select-option>
                        </a-select>
                      </div> -->
                      <!-- 专有算法的参数设置 -->
                      <div v-if="String(item.label_display).indexOf('增值服务组件') != -1">
                        <el-form>
                          <el-form-item :label="item.label_display" prop="algorithmType">

                            <el-select v-if="item.label_display != '增值服务组件（无量纲化）'"
                                       v-model="item.parameters[item.use_algorithm]"
                                       placeholder="请选择算已上传的私有算法" :teleported="false" style="width: 250px;">
                              <el-option v-for="item in privateAlgorithmList"
                                         :key="item.label"
                                         :label="item.label"
                                         :value="item.label"
                                         style="width: 200px; height: auto; background-color: white;"/>
                            </el-select>

                            <!-- 无量纲化的专有算法相较于其他不需要额外参数设置的算法要进行更多参数设定，因此绑定的参数也不一样 -->
                            <el-select v-if="item.label_display == '增值服务组件（无量纲化）'"
                                       v-model="item.parameters[item.use_algorithm]['algorithm']"
                                       placeholder="请选择算已上传的私有算法" :teleported="false" style="width: 250px;">
                              <el-option v-for="item in privateAlgorithmList"
                                         :key="item.label"
                                         :label="item.label"
                                         :value="item.label"
                                         style="width: 200px; height: auto; background-color: white;"/>
                            </el-select>
                          </el-form-item>
                        </el-form>
                        <!-- 专有算法的算法描述 -->
                        <div>
                          <span>算法描述：</span>
                          <div>{{ extraAlgorithmStatement[item.parameters[item.use_algorithm]] }}</div>
                        </div>
                      </div>
                      <template #reference>

                        <!-- 建模区中算法节点的形状 -->
                        <!-- <el-tooltip effect="dark" content="通过节点中心的附着点拖拽节点，或右击打开参数配置" :teleported="false"> -->
                          <div class="node-info-label font-style: italic;" :id=item.id>
                            <!-- <span>{{ item.label_display }}</span> -->
                            <!-- 节点的拖拽动作的识别点 -->
                            <img :src="setIconOfAlgorithms(item.label)" alt="icon" width="50px" height="50px"/>
                            <div style="
                            position: absolute; left: 88px; top: 50%; width: 6px; height: 6px;
                            border: 2px solid #80a5ba; 
                            border-radius: 50%; 
                            background-color: transparent;
                            cursor: move;" @mouseup="handleMouseup($event, item)"></div>
                            <div class="module-name" :title="item.label_display">{{ item.label_display }}</div>
                            <!-- 删除节点的删除按钮 -->
                            <el-button type="danger" icon="Delete" circle size="small" class="deleteButton"
                                       @click="deleteNode(item.nodeId)" :disabled="modelSetup"/>
                          </div>
                        <!-- </el-tooltip> -->
                      </template>
                    </el-popover>
                  </span>
                <!-- </el-tooltip> -->
                <!-- @contextmenu="params_setting(item[parameters])" -->
                <!-- 节点上的连线附着点 -->
                <div class="node-drag" :id="item.id"></div>
              </Vue3DraggableResizable>
            </DraggableContainer>
            <!-- <div style="width: 1000px; height: 100px; background-color: #88b6fb;"></div> -->
            <!-- <div
              style="position: absolute; right: 250px; bottom: 10px; width: 600px; height: auto;display: flex; justify-content: space-between; align-items: center;">
         
              <el-space size="large">

                <el-button type="primary" round style="width: 125px; font-size: 17px; "
                  @click="handleClear" icon="Refresh" class="operation-button">
                  清空模型
                </el-button>

                <el-button v-if="!toRectifyModel" type="primary" :disabled="canCompleteModeling"
                  round style="width: 125px; font-size: 17px;" @click="finishModeling"
                  icon="Check" class="operation-button">
                  完成建模
                </el-button>
              
                <el-button v-if="toRectifyModel" type="primary" :disabled="canCompleteModeling"
                  round style="width: 125px; font-size: 17px;" @click="rectifyModel"
                  icon="Edit" class="operation-button">
                  修改模型
                </el-button>

                
                <el-button type="primary" :disabled="canCheckModel" @mouseover="checkModeling" round
                  style="width: 125px; font-size: 17px; " @click="checkModel" icon="Search" class="operation-button">
                  检查模型
                </el-button>

           
                <el-button type="primary" :disabled="canSaveModel" @mouseover="saveModeling" round
                  style="width: 125px; font-size: 17px;" @click="saveModelSetting(true, [])" icon="Finished" class="operation-button">
                  保存模型
                </el-button>
              
                <el-button type="success" round style="width: 125px; font-size: 17px; " @click="startProgram"
                  icon="SwitchButton" :disabled="canStartProcess || processIsShutdown" @mouseover="startModeling" class="operation-button">
                  开始运行
                </el-button>
                <el-button :disabled="canShutdown" type="danger" round style="width: 125px; font-size: 17px;"
                  @click="shutDown" icon="Close" class="operation-button">
                  终止运行
                </el-button>
              </el-space>

            </div> -->
            <div
                style="position: absolute; right: 250px; bottom: 10px; width: 600px; height: auto;display: flex; justify-content: space-between; align-items: center;">

              <el-space size="large">

                <el-button type="primary" style="width: 125px; font-size: 17px; "
                           @click="handleClear" icon="Refresh" class="operation-button">
                  清空模型
                </el-button>

                <el-button v-if="!toRectifyModel" type="primary" :disabled="canCompleteModeling"
                           style="width: 125px; font-size: 17px;" @click="finishModeling"
                           icon="Check" class="operation-button">
                  完成建模
                </el-button>

                <el-button v-if="toRectifyModel" type="primary" :disabled="canCompleteModeling"
                           style="width: 125px; font-size: 17px;" @click="rectifyModel"
                           icon="Edit" class="operation-button">
                  修改模型
                </el-button>


                <el-button type="primary" :disabled="canCheckModel" @mouseover="checkModeling"
                           style="width: 125px; font-size: 17px; " @click="checkModel" icon="Search"
                           class="operation-button">
                  检查模型
                </el-button>


                <el-button type="primary" :disabled="canSaveModel" @mouseover="saveModeling"
                           style="width: 125px; font-size: 17px;" @click="saveModelSetting(true, [])" icon="Finished"
                           class="operation-button">
                  保存模型
                </el-button>

                <el-button type="success" style="width: 125px; font-size: 17px; " @click="startProgram"
                           icon="SwitchButton" :disabled="canStartProcess || processIsShutdown"
                           @mouseover="startModeling" class="operation-button">
                  开始运行
                </el-button>
                <el-button :disabled="canShutdown" type="danger" style="width: 125px; font-size: 17px;"
                           @click="shutDown" icon="Close" class="operation-button">
                  终止运行
                </el-button>
              </el-space>

            </div>
          </div>
          <!-- 模型运行结果展示区 -->
          <div class="resultsContainer"
               style="background-color: white;padding-bottom: 5px;">
            <!-- 显示程序运行的进度条 -->
            <div v-if="processing"
                 style="display: flex; justify-content: center; align-items: center; padding-top: 120px;">
              <span style="font-weight: 700; font-size: 22px">程序正在运行中</span>
              <el-progress :percentage="percentage" :indeterminate="true"/>
            </div>

            <!-- 点击在可视化建模区展示算法的具体介绍 -->
            <div style="width: 100%; height: 100%; background-color: white;"
                 v-if="(showPlainIntroduction || showStatusMessage) && !processing">
              <el-scrollbar height="480px" style="background-color: white;">
                <a-button type="text" style="position: absolute; top: 5px; right: 5px" v-if="showPlainIntroduction"
                          @click="showPlainIntroduction = false">关闭
                </a-button>
                <v-md-preview v-if="showPlainIntroduction" :text="introductionToShow"
                              style="text-align: left;"></v-md-preview>
                <v-md-preview v-if="showStatusMessage" :text="statusMessageToShow"
                              style="text-align: center; padding-top: 80px"></v-md-preview>
              </el-scrollbar>
            </div>

            <!-- 结果可视化区默认为自定义建模或是预定义模型的使用介绍 -->
            <div
                v-if="!showPlainIntroduction && !showStatusMessage && !canShowResults && !contrastVisible && !processing"
                style="background-color: white; height: 100%; width: auto">
              <el-scrollbar height="100%">
                <!-- 自定义建模 -->
                <div v-if="(userRole==='superuser' || userRole === 'user') && modelSelection === 'customModel'">
                  <v-md-preview :text="howToCustomizeModel" style="text-align: left;"></v-md-preview>
                  <div style="text-align: left; padding-left: 40px;">
                    <h4>1.机器学习的故障诊断流程推荐</h4>
                    <img src="../assets/customize-model-1.png" width="900px" alt="">
                    <div>模版用例：
                      <a-button @click="useModel(predefinedModel['templateModel1'])">用例1</a-button>
                      <a-button>用例2</a-button>
                    </div>
                    <br>
                    <h4>1.深度学习的故障诊断流程推荐</h4>
                    <img src="../assets/customize-model-2.png" width="600px" alt="">
                    <div>模版用例：
                      <a-button @click="useModel(predefinedModel['templateModel1'])">用例1</a-button>
                      <a-button>用例2</a-button>
                    </div>
                  </div>
                </div>
                <!-- 预定义模型 -->
                <div v-if="userRole === 'superuser' && modelSelection === 'templateModel'">
                  <v-md-preview :text="howToUseTemplateModel" style="text-align: left;"></v-md-preview>
                </div>
                <div v-if="userRole === 'user' && modelSelection === 'templateModel'">
                  <v-md-preview :text="howToUseTemplateModel2" style="text-align: left;"></v-md-preview>
                </div>
              </el-scrollbar>
            </div>

            <!-- 当点击二级目录后，展示二级目录下各个算法的优劣比较 -->
            <div v-if="contrastVisible && !processing"
                 style="background-color: white; height: 100%; width: auto; position: relative;">

              <el-scrollbar height="570px" style="background-color: white;">
                <a-button type="text" style="position: absolute; top: 5px; right: 5px" @click="contrastVisible = false">
                  关闭
                </a-button>
                <v-md-preview :text="contrastToShow" style="text-align: left;"></v-md-preview>
              </el-scrollbar>
            </div>

            <!-- 结果可视化各组件的结果展示 -->
            <el-scrollbar height="600px" v-if="canShowResults && !processing" style="background-color: white;">

              <!-- 用户反馈对话框 -->
              <div style="width: 100%; position: relative; z-index: 999;">
                <span style="position: absolute; right: 30px; top: 55px;">
                  <a-button circle :icon="h(EditOutlined)"
                  @click="feedBackDialogVisible = true" ></a-button> 反馈
                </span>
                <a-modal v-model:open="feedBackDialogVisible" title="用户反馈" cancelText="取消" okText="提交" @ok="feedBack">
                  <div style="font-weight: 600; font-size: 14px; margin-bottom: 20px">
                    <p>当前使用的模型：{{ modelLoaded }}</p>
                    <p>当前使用的数据：{{ usingDatafile }}</p>
                  </div>
                  <a-form :model="feedBackFormRefState" :rules="feedBackRules" ref="feedBackFormRef">
                    <a-form-item label="当前模型中存在疑问的模块" name="module">
                      <a-select style="width: 70%" placeholder="请选择组件" v-model:value="feedBackFormRefState.module">
                        <a-select-option v-for="item in contentJson.modules" :value="item">
                          {{ item }}
                        </a-select-option>
                      </a-select>
                    </a-form-item>
                    <a-form-item name="feedbackContent">
                      <div>问题描述</div>
                      <a-input style="width: 80%" v-model:value="feedBackFormRefState.feedbackContent"/>
                    </a-form-item>
                    <!-- <a-form-item>
                      <a-button @click="feedBack"></a-button>
                    </a-form-item> -->
                  </a-form>
                </a-modal>
              </div>
              <!-- 健康评估可视化 -->
              <!-- 不同样本的评估结果 -->
              <el-tabs type="border-card" tab-position="top" v-model="healthEvaluationOfExample"
                       v-if="displayHealthEvaluation">
                <el-tab-pane label="总结论" name="总结论">
                  <div style="display:flex; flex-direction: row; align-items: center;">
                    <div id="healthEvaluationPieChart" style="width: 600px; height: 500px"></div>
                    <div style="width: 700px;">
                      <el-text style="font-weight: bold; font-size: 18px;">{{
                          finalSuggestion
                        }}
                      </el-text>
                    </div>
                    <!-- 绘制饼状图 -->
                  </div>
                </el-tab-pane>

                <el-tab-pane v-for="(value, key) in resultsBarOfAllExamples" :key="key" :label="key" :name="key">
                  <el-tabs type="border-card" tab-position="left" v-model="activeName1">
                    <el-tab-pane label="层级有效指标" name="first">
                      <!-- <img :src="healthEvaluationFigure1" alt="figure1" id="health_evaluation_figure_1"
                        class="result_image" style="width: auto; height: 450px;" /> -->
                      <el-image
                          style="width: auto; height: 450px;"
                          :src="resultsBarOfAllExamples[key]"
                          :zoom-rate="1.2"
                          :max-scale="7"
                          :min-scale="0.2"
                          :preview-src-list="[resultsBarOfAllExamples[key]]"
                          :initial-index="4"
                          fit="cover"
                      />
                    </el-tab-pane>
                    <el-tab-pane label="指标权重" name="second">
                      <!-- <img :src="healthEvaluationFigure2" alt="figure2" id="health_evaluation_figure_2"
                        class="result_image" style="width: auto; height: 450px;" /> -->
                      <el-image
                          style="width: auto; height: 450px;"
                          :src="levelIndicatorsOfAllExamples[key]"
                          :zoom-rate="1.2"
                          :max-scale="7"
                          :min-scale="0.2"
                          :preview-src-list="[levelIndicatorsOfAllExamples[key]]"
                          :initial-index="4"
                          fit="cover"
                      />
                    </el-tab-pane>
                    <el-tab-pane label="评估结果" name="third">
                      <div
                          style="display:flex; flex-direction: column; justify-content: center; align-items: center; width: 100%; height: 100%">
                        <el-image
                            style="width: 800px; height: auto;"
                            :src="statusOfExamples[key]"
                            :zoom-rate="1.2"
                            :max-scale="7"
                            :min-scale="0.2"
                            :preview-src-list="[statusOfExamples[key]]"
                            :initial-index="4"
                            fit="cover"
                        />
                        <br>

                        <div style="width: 700px;">
                          <el-text :v-model="suggestionOfAllExamples[key]" style="font-weight: bold; font-size: 18px;">
                            {{
                              suggestionOfAllExamples[key]
                            }}
                          </el-text>
                        </div>
                      </div>
                      <!-- <img :src="healthEvaluationFigure3" alt="figure3" id="health_evaluation_figure_3"
                        class="result_image" style="width: auto; height: 360px;" /> -->


                    </el-tab-pane>
                  </el-tabs>
                </el-tab-pane>

              </el-tabs>

              <!-- 特征提取可视化 -->
              <div v-if="displayFeatureExtraction" style="justify-content: center;">
                <el-tabs tab-position="left" type="border-card" v-model="featuresExtractionRawData">
                  <el-tab-pane v-for="item in rawDataList" :key="item.snesor_no" :label="item.sensor_no"
                               :name="item.sensor_no">
                    <div :id="item.sensor_no" style="width: 1300px; height: 400px"></div>
                    <!-- <div style="padding-left: 10px;text-align: left; font-size: 25px; color:darkgrey;">由原始信号提取特征：</div> -->
                    <!-- 对应特征提取结果 -->
                    <div :id="item.sensor_no + 'features'" style="width: 1300px; height: 400px"></div>
                  </el-tab-pane>
                </el-tabs>
                <!-- <div id="rawDataFigure" style="width: 900px; height: 400px"></div> -->
                <!-- <el-table :data="transformedData" style="width: 96%; margin-top: 20px;"
                 >
                  <el-table-column v-for="column in columns" :key="column.prop" :prop="column.prop" :label="column.label"
                  >
                  </el-table-column>
                </el-table> -->
              </div>

              <!-- 特征选择可视化 -->
              <div v-if="displayFeatureSelection">
                <el-tabs v-model="featuresSelectionTabs">
                  <el-tab-pane label="特征选择结果" name="first">
                    <el-scrollbar height="480px">
                      <!-- <img :src="featureSelectionFigure" alt="feature_selection_figure1" class="result_image"
                        style="width: 900px; height: 430px;" /> -->
                      <el-image
                          style="width: auto; height: 430px;"
                          :src="featureSelectionFigure"
                          :zoom-rate="1.2"
                          :max-scale="7"
                          :min-scale="0.2"
                          :preview-src-list="[featureSelectionFigure]"
                          :initial-index="4"
                          fit="cover"
                      />
                      <br>
                      <div style="width: 1000px; margin-left: 250px;">
                        <span style="font-weight: bold; font-size: 15px;">根据规则：{{ selectFeatureRule }}，选取特征结果为： {{
                            featuresSelected
                          }}</span>
                      </div>
                    </el-scrollbar>

                  </el-tab-pane>

                  <el-tab-pane label="相关系数矩阵热力图" name="second">
                    <el-scrollbar height="480px">
                      <!-- <img :src="correlationFigure" alt="correlation_figure" class="result_image"
                      style="width: auto; height: 500px;" /> -->
                      <el-image
                          style="width: auto; height: 500px;"
                          :src="correlationFigure"
                          :zoom-rate="1.2"
                          :max-scale="7"
                          :min-scale="0.2"
                          :preview-src-list="[correlationFigure]"
                          :initial-index="4"
                          fit="cover"
                      />
                    </el-scrollbar>

                  </el-tab-pane>

                </el-tabs>

              </div>

              
              <!-- 故障诊断可视化 -->
              <div v-if="displayFaultDiagnosis" class="result-visualization-container">
                <!-- <div style="font-weight: bold;">
                  故障诊断结果为： 由输入的振动信号，根据故障诊断算法得知该部件<span :v-model="faultDiagnosis"
                    style="font-weight: bold; color: red;">{{
                      faultDiagnosis }}</span>
                </div> -->
                <v-md-preview style="padding: 0px; margin: 0px" :text="faultDiagnosisResultsText"></v-md-preview>
                
                <!-- 用户填写反馈的对话框 -->
                
                <el-tabs v-model="faultDiagnosisResultOption" tab-position="top">
                  <el-tab-pane key="1" label="连续样本指标变化">
                    <!-- 连续样本指标变化的折线图 -->
                    <div id="indicatorVaryingFigure" style="width: 1200px; height: 500px"></div>
                  </el-tab-pane>
                  <el-tab-pane key="2" label="不同类型样本占比">
                    <!-- 故障样本与非故障样本数量饼状图 -->
                    <div id="faultExampleRatioFigure" style="width: 1200px; height: 500px"></div>
                  </el-tab-pane>
                  <el-tab-pane key="3" label="原始信号波形图">
                    <div style="width: 1200px; height: 500px">
                      <el-image
                          style="width: auto; height: 450px;"
                          :src="faultDiagnosisFigure"
                          :zoom-rate="1.2"
                          :max-scale="7"
                          :min-scale="0.2"
                          :preview-src-list="[faultDiagnosisFigure]"
                          :initial-index="4"
                          fit="cover"
                      />
                    </div>

                  </el-tab-pane>
                </el-tabs>

                <!-- <img :src="faultDiagnosisFigure" alt="fault_diagnosis_figure" class="result_image"
                  style="width: auto; height: 450px;" /> -->

              </div>
              <!-- 故障故障预测可视化 -->
              <div v-if="displayFaultRegression" style="margin-top: 20px; font-size: 18px;">
                <div style="width: 1000px; margin-left: 250px;  font-weight: bold">
                  经故障诊断算法，目前该部件<span :v-model="faultRegression" style="font-weight: bold; color: red;">{{
                    faultRegression
                  }}</span>
                  <span v-if="!haveFault" :v-model="timeToFault" style="font-weight: bold;">根据故障预测算法预测，该部件{{
                      timeToFault
                    }}后会出现故障</span>
                </div>
                <br>
                <!-- <img :src="faultRegressionFigure" alt="fault_regression_figure" class="result_image"
                  style="width: auto; height: 450px;" /> -->
                <el-image
                    style="width: auto; height: 450px;"
                    :src="faultRegressionFigure"
                    :zoom-rate="1.2"
                    :max-scale="7"
                    :min-scale="0.2"
                    :preview-src-list="[faultRegressionFigure]"
                    :initial-index="4"
                    fit="cover"
                />
              </div>
              <!-- 插值处理结果可视化 -->
              <!-- <div v-if="displayInterpolation" style="margin-top: 20px; font-size: 18px;">
                <br>
                <img :src="interpolationFigure" alt="interpolation_figure" class="result_image"
                  style="width: 900px; height: 450px;" />
              </div> -->
              <el-tabs v-model="activeName3" v-if="displayInterpolation" type="border-card">
                <el-tab-pane v-for="item in interpolationResultsOfSensors" :key="item.name" :label="item.label"
                             :name="item.name">
                  <!-- <img :src="interpolationFigures[item.name - 1]" alt="figure" id="figure"
                  class="result_image" style="width: 900px; height: 450px;" /> -->
                  <el-image
                      style="width: auto; height: 450px;"
                      :src="interpolationFigures[item.name - 1]"
                      :zoom-rate="1.2"
                      :max-scale="7"
                      :min-scale="0.2"
                      :preview-src-list="[interpolationFigures[item.name - 1]]"
                      :initial-index="4"
                      fit="cover"
                  />
                </el-tab-pane>
              </el-tabs>
              <!-- 无量纲化可视化 -->
              <div v-if="displayNormalization" style="font-size: 18px;">
                <!-- 针对提取到的特征进行的无量纲化得到的结果 -->
                <div v-if="normalizationResultType == 'table'">
                  <div style="font-size: large;">原数据</div>
                  <el-table :data="normalizationFormdataRaw" style="width: 96%; margin-top: 20px;"
                  >
                    <el-table-column v-for="column in normalizationColumns" :key="column.prop" :prop="column.prop"
                                     :label="column.label"
                                     :width="column.width">
                    </el-table-column>
                  </el-table>
                  <br>
                  <div style="font-size: large;">标准化后数据</div>
                  <el-table :data="normalizationFormdataScaled" style="width: 96%; margin-top: 20px;"
                  >
                    <el-table-column v-for="column in normalizationColumns" :key="column.prop" :prop="column.prop"
                                     :label="column.label"
                                     :width="column.width">
                    </el-table-column>
                  </el-table>
                </div>
                <!-- 针对原始信号序列进行无量纲化得到的结果 -->
                <el-tabs v-model="activeName4" v-if="normalizationResultType == 'figure'" type="border-card">
                  <el-tab-pane v-for="item in normalizationResultsSensors" :key="item.name" :label="item.label"
                               :name="item.name">
                    <!-- <img :src="normalizationResultFigures[item.name - 1]" alt="figureOfSensor" id="figure"
                    class="result_image" style="width: 900px; height: 450px;" /> -->
                    <el-image
                        style="width: 900; height: 450px;"
                        :src="normalizationResultFigures[item.name - 1]"
                        :zoom-rate="1.2"
                        :max-scale="7"
                        :min-scale="0.2"
                        :preview-src-list="[normalizationResultFigures[item.name - 1]]"
                        :initial-index="4"
                        fit="cover"
                    />
                  </el-tab-pane>
                </el-tabs>

              </div>
              <!-- 小波降噪可视化 -->

              <!-- <img :src="denoiseFigure" alt="denoise_figure" class="result_image"
                style="width: 900px; height: 450px;" /> -->
              <el-tabs v-model="activeName2" v-if="displayDenoise" type="border-card">
                <el-tab-pane v-for="item in waveletResultsOfSensors" :key="item.name" :label="item.label"
                             :name="item.name">
                  <!-- <img :src="denoiseFigures[item.name - 1]" alt="figure" id="figure"
                  class="result_image" style="width: 900px; height: 450px;" /> -->
                  <el-image
                      style="width: auto; height: 450px;"
                      :src="denoiseFigures[item.name - 1]"
                      :zoom-rate="1.2"
                      :max-scale="7"
                      :min-scale="0.2"
                      :preview-src-list="[denoiseFigures[item.name - 1]]"
                      :initial-index="4"
                      fit="cover"
                  />
                </el-tab-pane>
              </el-tabs>

              <!-- 展示用户文件中原始数据 -->
              <div v-if="displayRawDataWaveform" style="padding: 20px; position: relative;">
                <el-button text
                           style="color: green;position: absolute; right: 10px; top: 10px; width: 90px; height: 50px"
                           @click="displayRawDataWaveform = false">关闭
                </el-button>
                <p style="text-align: left; color: red; font-weight: bolder; font-size: 18px">已上传数据文件内容浏览</p>
                <p style="font-size: 18px">文件名：{{ currentDataBrowsing }}</p>
                <el-image
                    style="width: auto; height: 450px;"
                    :src="rawDataWaveform"
                    :zoom-rate="1.2"
                    :max-scale="7"
                    :min-scale="0.2"
                    :preview-src-list="[rawDataWaveform]"
                    :initial-index="4"
                    fit="cover"
                />

              </div>

            </el-scrollbar>

          </div>

        </el-main>


        <!-- 以抽屉的形式打开用户历史模型 -->
        <el-drawer v-model="modelsDrawer" direction="ltr" size="35%">

          <div style="display: flex; flex-direction: column; ">
            <el-col>

              <h2 style=" margin-bottom: 25px; color: #253b45;">系统模型库</h2>
              <span style="font-size: 15px; color:#a1a2b1">*当前用户权限为{{
                  userRole === 'user' ? '普通用户，可以使用系统中的模型' : '系统用户，可以添加模型和删除本用户添加的模型'
                }}</span>
              <el-table :data="fetchedModelsInfo" height="500" stripe style="width: 100%">
                <el-table-column :width="100" property="author" label="模型作者"/>
                <el-table-column :width="200" property="model_name" label="模型名称" show-overflow-tooltip/>
                <el-table-column :width="260" label="操作">
                  <template #default="scope">
                    <el-button size="small" type="primary" style="width: 50px;" @click="useModel(scope.row)">
                      使用
                    </el-button>
                    <el-button size="small" type="danger" style="width: 50px;" v-if="userRole === 'superuser'"
                               @click="deleteModel(scope.$index, scope.row)">
                      删除
                    </el-button>
                    <el-popover placement="bottom" :width='500' trigger="click">
                      <el-descriptions :title="modelName" :column="3" direction="vertical"
                      >
                        <el-descriptions-item label="使用模块" :span="3">
                          <el-tag size="small" v-for="algorithm in modelAlgorithms">{{ algorithm }}</el-tag>
                        </el-descriptions-item>
                        <el-descriptions-item label="算法参数" :span="3">
                          <div v-for="item in modelParams">{{ item.模块名 }}: {{ item.算法 }}</div>
                        </el-descriptions-item>
                      </el-descriptions>
                      <template #reference>
                        <el-button size="small" type="info" style="width: 80px" @click="showModelInfo(scope.row)">
                          查看模型
                        </el-button>
                      </template>
                    </el-popover>
                  </template>
                </el-table-column>
              </el-table>

              <el-dialog v-model="deleteModelConfirmVisible" title="提示" width="500">
                <span style="font-size: 20px;">确定删除该模型吗？</span>
                <template #footer>
                  <el-button
                      style="width: 150px"
                      @click="deleteModelConfirmVisible = false"
                  >取消
                  </el-button
                  >
                  <el-button
                      style="width: 150px; margin-right: 70px"
                      type="primary"
                      @click="deleteModelConfirm"
                  >确定
                  </el-button
                  >
                </template>
              </el-dialog>
            </el-col>
          </div>
        </el-drawer>

        <!-- 以抽屉的形式打开用户历史数据 -->
        <el-drawer v-model="dataDrawer" direction="ltr" size="45%">
          <div style="display: flex; flex-direction: column">
            <el-col>
              <h2 style="margin-bottom: 25px; color: #253b45">用户数据文件</h2>

              <el-table :data="fetchedDataFiles" height="500" stripe style="width: 100%">
                <el-table-column :width="100" property="owner" label="文件上传者" show-overflow-tooltip/>
                <el-table-column :width="200" property="dataset_name" label="文件名称" show-overflow-tooltip/>
                <el-table-column :width="230" property="description" label="文件描述" show-overflow-tooltip/>
                <el-table-column :width="210" label="操作">
                  <template #default="scope">
                    <el-button
                        size="small"
                        type="primary"
                        style="width: 50px"
                        @click="useDataset(scope.row)"
                        :loading="loadingData"
                    >
                      使用
                    </el-button>
                    <el-popconfirm title="你确定要删除该数据文件吗" @confirm="deleteDatasetConfirm(scope.$index, scope.row)">
                      <template #reference>
                        <!-- <el-button
                          size="small"
                          type="danger"
                          style="width: 50px"
                        >
                          删除组件
                        </el-button> -->
                        <el-button
                            size="small"
                            type="danger"
                            style="width: 50px"
                        >
                          删除
                        </el-button>
                      </template>
                      <template #actions="{ confirm, cancel }">
                        <el-row>
                          <el-col :span="12">
                            <el-button size="small" @click="cancel">取消</el-button>
                          </el-col>
                          <el-col :span="12">
                            <el-button
                                type="primary"
                                size="small"
                                @click="confirm"
                            >
                              确定
                            </el-button>
                          </el-col>
                        </el-row>
                      </template>
                    </el-popconfirm>

                    <el-button
                        size="small"
                        type="success"
                        style="width: 50px"
                        @click="browseDataset(scope.row)"
                    >
                      查看
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>

              <!-- <el-dialog
                v-model="deleteDatasetConfirmVisible"
                title="提示"
                width="500"
              >
                <span style="font-size: 20px">确定删除该数据文件吗？</span>
                <template #footer>
                  <el-button
                    style="width: 150px"
                    @click="deleteDatasetConfirmVisible = false"
                    >取消</el-button
                  >
                  <el-button
                    style="width: 150px; margin-right: 70px"
                    type="primary"
                    @click="deleteDatasetConfirm"
                    >确定</el-button
                  >
                </template>
              </el-dialog> -->
            </el-col>

          </div>

        </el-drawer>

      </el-container>

    </el-container>
    <el-dialog v-model="dialogModle" title="保存模型" draggable width="30%">
      <el-form :model="modelInfoForm" :rules="rules">
        <el-form-item label="模型名称" :label-width='140' prop="name"
        >
          <el-input style="width: 160px;" v-model="modelInfoForm.name" autocomplete="off"/>
        </el-form-item>
      </el-form>
      <span class="dialog-footer">
        <el-button style="margin-left: 85px; width: 150px;" @click="dialogModle = false">取消</el-button>
        <el-button style="width: 150px;" type="primary" @click="saveModelConfirm">确定</el-button>
      </span>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>

import {onMounted, nextTick, ref, h, reactive} from 'vue'
import {jsPlumb} from 'jsplumb'
import {Action, ElNotification, ElMessage, ElMessageBox} from "element-plus";
import axios from 'axios';
import {DraggableContainer} from "@v3e/vue3-draggable-resizable";
import {computed} from 'vue';
import {useRouter} from 'vue-router';
import api from '../utils/api.js'
import {
  labelsForAlgorithms,
  plainIntroduction,
  labelsForParams,
  contrastOfAlgorithm,
  predefinedModel
} from '../components/constant.ts'
import * as echarts from 'echarts';
import {EditOutlined, UploadOutlined} from "@ant-design/icons-vue";
import {message} from "ant-design-vue";
import type {UploadProps} from "ant-design-vue";
import {Rule} from "ant-design-vue/es/form";
import uploadPrivateAlgorithmFiless from '../components/uploadPrivateAlgorithmFiles.vue'
import managePrivateAlgorithm from '../components/managePrivateAlgorithm.vue';
import {FolderOutlined, FolderOpenOutlined, QuestionCircleOutlined} from '@ant-design/icons-vue';
import {pa} from 'element-plus/es/locales.mjs';
import {multipleCascaderProps} from 'ant-design-vue/es/vc-cascader/Cascader';

// 删除增值服务组件时需要刷新增值组件列表
const handleDeleteExtraModule = ()=>{
  getExtraAlgorithmMao();
}
// 上传增值服务组件时需要刷新增值组件列表
const handleAddExtraModule = ()=>{
  getExtraAlgorithmMao();
}

//控制增值组件目录开关
const isShowSecondButton = ref(false);
const fetchedExtraAlgorithmList = ref([])

//构造数据
const options_modules = ref([
  {
    label: '插值处理', id: '1.1', use_algorithm: null, alias:null, machineLearning: '', parameters: {
      'private_interpolation': '',
    }, tip_show: false, tip: '使用专有插值处理方法', optional: false
  } ,
  {label: '特征提取', id: '1.2', use_algorithm: null, alias:null, machineLearning: '', parameters: {
      'private_feature_extraction':'',
    }},
  {
    label: '无量纲化', id: '1.5', use_algorithm: null, alias:null, machineLearning: '', parameters: {
      'private_scaler': {useLog: false, algorithm: ''}
    }, tip_show: false, tip: '使用专有无量纲化处理方法', optional: true
  },
  {
    label: '特征选择', id: '1.3', use_algorithm: null, alias:null, machineLearning: '', parameters: {
      'extra_feature_selection': {rule: 1, threshold1: 0.1, threshold2: 0.1}
    }, tip_show: false, tip: '使用专有特征选择方法', optional: true
  },
  {
    label: '小波变换', id: '1.4', use_algorithm: null, alias:null, machineLearning: '', parameters: {
      'extra_wavelet_transform': ''
    }, tip_show: false, tip: '对输入信号进行小波变换', optional: true
  },
  {
    label: '故障诊断', id: '2.1', use_algorithm: null, alias:null, machineLearning: '', parameters: {
      'private_fault_diagnosis_machine_learning': '',
      'private_fault_diagnosis_deeplearning': '',
    }, tip_show: false, tip: '使用专有故障诊断方法', optional: false
  },
  {
    label: '故障预测', id: '2.2', use_algorithm: null, alias:null, machineLearning: '', parameters: {

      'private_fault_prediction': {}
    }, tip_show: false, tip: '使用专有故障预测方法', optional: false
  },
  {
    label: '专有健康评估', id: '3.4', use_algorithm: null, alias:null, machineLearning: '', parameters: {
      'extra_health_evaluation': ''
    }, tip_show: false, tip: '使用专有健康评估的评价方法', optional: false, 
  },

])


//构造增值组件菜单,复制速度不能过快，foreach和map太快会直接覆盖到前面的内容
function updateOptionsWithBackendData(data) {
  // 清空 fetchedExtraAlgorithmList，准备存储新数据
  fetchedExtraAlgorithmList.value = [];

  // 使用 for 循环遍历 data 数组
  for (let i = 0; i < data.length; i++) {
    const item = data[i];
    // 在 options_modules 中查找与当前 item 的 algorithmType 匹配的选项
    const foundOption = options_modules.value.find(option => option.label === item.algorithmType);

    // 如果找到了匹配的选项
    if (foundOption) {
      // 创建一个新对象，它是 foundOption 的副本
      const newOption = {...foundOption};

      // 更新新对象的属性
      newOption.label = item.algorithmType
      // newOption.use_algorithm = item.algorithmName;
      // newOption.use_algorithm = newOption.parameters[]
      newOption.alias = item.alias;
      newOption.machineLearning = item.machineLearning;

      // 将新对象添加到 fetchedExtraAlgorithmList 中
      fetchedExtraAlgorithmList.value.push(newOption);
    }
  }

}


//获取用户的增值组件列表
const getExtraAlgorithmMao = () => {
  api.get("/user/user_fetch_extra_algorithm/").then((response: any) => {
    if (response.data.code == 401) {
      ElMessageBox.alert("登录状态失效，请重新登陆", "提示", {
        confirmButtonText: "确定",
        callback: () => {
          router.push("/");
        },
      });
    }
    if (response.data.code == 200) {
      console.log('信息：', response.data.data)
      updateOptionsWithBackendData(response.data.data)
    }
  });
};

//处理增值组件的拖拽
const handleDragendAdd = (ev, algorithm, node) => {
  // 使用find方法查找具有特定alias的对象
  console.log("现拖拽algorithm", algorithm)
  const foundObject = fetchedExtraAlgorithmList.value.find(obj => obj.alias === algorithm)
  // const parametersKey = Object.keys(foundObject.parameters)[0]
  const machineLearning = foundObject.machineLearning
  const parametersKey = machineLearning === 'ml' ? 'private_fault_diagnosis_machine_learning' : 'private_fault_diagnosis_deeplearning'
  console.log("参数的键",parametersKey)
  console.log("foundObject: ", foundObject)
  console.log("foundObject.parameters: ", foundObject.parameters)
  // 如果找到了对象，复制其parameters的键
  if (foundObject) {
    console.log("找到的算法文件", parametersKey); // 输出: ['private_fault_diagnosis_deeplearning', 'private_fault_diagnosis_machine_learning']
  }
  node.parameters[parametersKey] = node.alias
  console.log("node.parameters", node.parameters)
  // 拖拽进来相对于地址栏偏移量
  const evClientX = ev.clientX
  const evClientY = ev.clientY
  let left
  if (evClientX < 300) {
    left = evClientX + 'px'
  } else {
    left = evClientX - 300 + 'px'
  }
  console.log("node.label: ", node.label)
  let top = 50 + 'px'
  const nodeId = node.id
  const nodeInfo = {
    label_display: node.alias,   // 具体算法的名称

    label: node.label,      // 算法模块名称
    id: node.id,
    nodeId,
    nodeContainerStyle: {
      left: left,
      top: top,
    },
    use_algorithm: parametersKey,
    parameters: node.parameters,
    optional: node.optional
  }

  console.log(nodeInfo)
  // 针对时域或是频域特征给出不同的可选特征

  // console.log(nodeInfo)
  //算法模块不允许重复
  if (nodeList.value.length === 0) {
    nodeList.value.push(nodeInfo)
  } else {
    let isDuplicate = false;
    for (let i = 0; i < nodeList.value.length; i++) {
      let nod = nodeList.value[i];
      if (nod.id == node.id) {
        // window.alert('不允许出现重复模块');
        ElMessage({
          message: '不允许出现同一类别的算法',
          type: 'warning'
        })
        isDuplicate = true;
        break;
      }
    }
    // 向节点列表中添加新拖拽入可视化建模区中的模块
    if (!isDuplicate) {
      nodeList.value.push(nodeInfo);
      console.log('画布列表', nodeList.value)
    }

  }

  // 将节点初始化为可以连线的状态
  nextTick(() => {
    plumbIns.draggable(nodeId, {containment: "efContainer"})

    if (node.id < 4) {
      plumbIns.makeSource(nodeId, deff.jsplumbSourceOptions)
    }

    if (node.id == '4') {
      plumbIns.makeSource(nodeId, deff.jsplumbSourceOptions)
      return
    }

    plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)

  })
}

// 动态绑定选择基础组件和系统模型按钮的样式，使得其背景色动态改变
const getRadioButtonStyle = (value) => {
  const baseStyle = {
    width: '250px',
    border: 'none',
    borderRadius: 0,
    fontWeight: 'bolder',
    fontSize: '22px'
  };

  if (modelSelection.value === value) {
    return {
      ...baseStyle,
      color: 'white',
      'background-color': "#4fb0ff"
    };
  } else {
    return {
      ...baseStyle,
      color: '#558b48'
    };
  }
}


// 建模区中各个算法节点的图标url
const setIconOfAlgorithms = (label: string) => {

  let iconName;
  switch (label) {
    case '插值处理':
      iconName = 'interpolation-icon.svg'
      break
    case '特征提取':
      iconName = 'extraction-icon.svg'
      break
    case '无量纲化':
      iconName = 'normalization-icon.svg'
      break
    case '特征选择':
      iconName = 'feature-selection-icon.svg'
      break
    case '小波变换':
      iconName = 'wavelet-icon.svg'
      break
    case '数据源':
      iconName = 'data-source-icon.svg'
      break
    case '故障诊断':
      iconName = 'fault-diagnosis-icon.svg'
      break
    case '故障预测':
      iconName = 'fault-prediction-icon.svg'
      break
    case '自定义模块':
      iconName = 'custom-module-icon.svg'
      break
    case '层次分析模糊综合评估':
    case '层次逻辑回归评估':
    case '层次朴素贝叶斯评估':
    case '健康评估':
      iconName = 'health-evaluation-icon.svg'
  }
  return new URL(`../assets/${iconName}`, import.meta.url).href
}

// 关于如何自定义建模的介绍
// const showHowToCustomizeModel = ref(true);
const howToCustomizeModel = "### 如何自定义建模？ \n " +
    "#### 1. 点击左侧菜单栏中的“基础组件”，在基础组件菜单下，可以选择任意组件拖入建模区, 右键点击建模区中的节点可进行相关参数配置。 \n" +
    "#### 2. 通过建模区中算法节点的附着点可进行算法模块间的连接 \n" +
    "#### 3. 建立模型时，还需包括数据源组件，将左侧的数据源组件拖入建模区，连接至模型的开始处，并且右键点击数据源组件可以进行数据的上传和加载操作。 \n" +
    "### 推荐的模型流程 \n"
// "#### 1. 机器学习的故障诊断 \n " +
// "<img src='./src/assets/customize-model-1.png' alt='推荐模型1' width='1000px'></img> \n" +
// "#### 2. 深度学习的故障诊断 \n " +
// "![推荐模型2](../assets/recommend-model-2.png '深度学习的故障诊断')"


// 关于如何使用预定义的模型和加载已经保存的模型的介绍
const howToUseTemplateModel = "### 如何使用预定义的模型？ \n " +
    "#### 1. 点击左侧菜单栏中顶部的“系统模型库”，进入模型加载区。 \n" +
    "#### 2. 点击菜单栏中的“打开模型库”按钮，打开系统模型库界面。 \n" +
    "#### 3. 在系统模型库中，包含所有系统用户开发并保存的模型，可以直接使用，同时可以对本系统用户创建的模型进行管理。 \n"

const howToUseTemplateModel2 = "### 如何使用预定义的模型？ \n " +
    "#### 1. 点击菜单栏中的“打开模型库”按钮，打开系统模型库界面。 \n" +
    "#### 2. 在系统模型库中，包含所有系统用户开发并保存的模型，作为普通用户可以直接使用。 \n"


// 确认上传文件对话框
const uploadconfirmLoading = ref<boolean>(false);
const uploadConfirmDialog = ref<boolean>(false);
// 确保弹出的确认上传文件对话框位于最顶层
// const getContainer = () => document.body;

const dataFileFormState = ref({
  filename: "",
  description: "",
  multipleSensors: 'single',
  isPublic: 'private',
});
const fileUploadFormRef = ref();
const fileNameRules: Record<string, Rule[]> = {
  filename: [
    {required: true, message: "请输入文件名", trigger: "blur"},
    // { pattern: /[<>:"\/\\|?*]/, message: '文件名中包含非法字符', trigger: 'blur' }
    {pattern: /^[\u4e00-\u9fa5_a-zA-Z0-9]+$/, message: '请输入中英文/数字/下划线', trigger: 'blur'},
    // { validator: validateFilename, trigger: 'blur' }
  ],
  description: [
    {required: true, message: "请输入文件描述", trigger: "blur"},
  ],
  multipleSensors: [
    {required: true, message: "请选择是否为多传感器数据", trigger: "blur"},
  ],
}


// 文件名不能为'无'
// function validateFilename(rule, value, callback) {
//   if (value === '无') {
//     callback(new Error('文件名不能为“无”'));
//   } else {
//     callback();
//   }
// }


// 确认上传文件
const onFinish = () => {

  uploadconfirmLoading.value = true;
  const formData = new FormData();
  formData.append("file", fileList.value[0]);
  formData.append("filename", dataFileFormState.value.filename);
  formData.append("description", dataFileFormState.value.description);
  formData.append("multipleSensors", dataFileFormState.value.multipleSensors);
  formData.append("isPublic", dataFileFormState.value.isPublic);
  uploading.value = true;

  api.post("/user/upload_datafile/", formData)
      .then((response: any) => {
        if (response.data.message == 'save data success') {
          fileList.value = [];
          uploading.value = false;
          message.success("数据文件上传成功");

          uploadconfirmLoading.value = false;
          uploadConfirmDialog.value = false;
        } else {
          uploading.value = false;
          message.error("文件上传失败, " + response.data.message);
          uploadconfirmLoading.value = false;
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
        uploadconfirmLoading.value = false;
        message.error("上传失败, " + error);
      });

};


const fileList = ref<UploadProps["fileList"]>([]);  // 文件列表
const uploading = ref<boolean>(false);
const loadingDataModel = ref<number>(1)


// 移除文件列表中的文件
const handleRemove: UploadProps["onRemove"] = (file) => {
  const index = fileList.value.indexOf(file);
  const newFileList = fileList.value.slice();
  newFileList.splice(index, 1);
  fileList.value = newFileList;
};


const beforeUpload: UploadProps["beforeUpload"] = (file) => {
  const allowedTypes = ['.npy', '.mat'];
  const fileType = file.name.split('.').pop().toLowerCase();
  if (!allowedTypes.includes('.' + fileType)) {
    ElMessage({
      message: '文件格式错误，请上传mat或npy文件',
      type: 'error',
    });
    return false;
  }

  fileList.value.length = 0;
  fileList.value = [...(fileList.value || []), file];
  return false;
};


// 文件类型检查，只允许mat或是npy格式的文件
// const handleUploadDataFile = () => {
//   let file = fileList.value[0]
//   let filename = file.name
//   const ext = filename.split('.').pop().toLowerCase();  
//   if (ext != 'mat' && ext != 'npy') {
//     ElMessage({
//       message: '文件格式错误，请上传mat或npy文件',
//       type: 'error',
//     })
//     return  
//   }
//   uploadConfirmDialog.value = true

// };


// 子组件向父组件发送数据
const emit = defineEmits(["switchDrawer"]);
const switchDrawer = () => {
  let url = 'user/fetch_datafiles/'
  api.get(url)
      .then((response: any) => {
        let datasetInfo = response.data
        modelsDrawer.value = false;

        fetchedDataFiles.value = []

        for (let item of datasetInfo) {
          fetchedDataFiles.value.push(item)
        }

        dataDrawer.value = true
        // emit("switchDrawer", fetchedDatasetsInfo);
      })

  // fetchData.forEach(element => {
  //   fetchedDataFiles.value.push(element)
  // });

};


const operationHelpDialogVisible = ref(false)  // 用户操作指南对话框
const userHelpDialogVisible = ref(false)       // 用户使用教程对话框
const userHelpDialogScrollbar = ref(null)
// 用户权限
const modelSelection = ref("customModel");

// 私有算法列表
const privateAlgorithmList = ref([])

const extraAlgorithmStatement = ref({})


// 额外算法的描述
// const extraAlgorithmStatement = ref('')

const getExtraAlgorithm = (item: any) => {

  let algorithmTypeMapping = {
    '插值处理': "private_interpolation", '特征选择': 'extra_feature_selection',
    '特征提取': 'private_feature_extraction', '无量纲化': 'private_scale',
    '小波变换': 'extra_wavelet_transform', '故障诊断': 'private_fault_diagnosis',
    '故障预测': 'private_fault_prediction', '健康评估': 'extra_health_evaluation'
  }


  //获取私有算法列表
  // 如果是故障诊断的私有算法，则需要根据具体是机器学习的故障诊断还是深度学习的故障诊断进行分类
  var algorithmType  // 私有算法类型
  if (item.label != '故障诊断') {
    algorithmType = algorithmTypeMapping[item.label]
  } else {
    if (item.label_display.indexOf('机器学习') != -1) {
      algorithmType = 'private_fault_diagnosis_ml'  //机器学习算法的故障诊断
    } else {
      algorithmType = 'private_fault_diagnosis_dl'  //深度学习算法的故障诊断
    }
  }

  api.get('/user/user_fetch_private_algorithm?algorithm_type=' + algorithmType).then((response) => {
    if (response.data.code == 200) {
      // 将字符串数组转换为对象数组
      const algorithmList = response.data.algorithmList
      console.log("algorithmList: ", algorithmList)
      const algorithms = algorithmList.map(item => ({label: item.algorithmAlias}))
      // extraAlgorithmStatement.value = algorithmList.map(item => ( { 'item.algorithmAlias': item.algorithmStatement }))

      algorithmList.forEach(element => {
        extraAlgorithmStatement.value[element.algorithmAlias] = element.algorithmStatement
      });

      console.log(extraAlgorithmStatement.value)
      privateAlgorithmList.value.length = 0
      privateAlgorithmList.value = algorithms

    }
    if (response.data.code == 401) {
      ElMessageBox.alert('登录状态已失效，请重新登陆', '提示',
          {
            confirmButtonText: '确定',
            callback: (action: Action) => {
              router.push('/')
            }
          }
      )
    }

  })
      .catch(error => {
        ElMessage({
          message: '获取私有算法列表失败,' + error,
          type: 'error'
        })
      })
}


// 保存模型时提交的模型名的规则验证
const rules = {
  name: [
    {required: true, message: "请输入模型名", trigger: "blur"},
    // { pattern:/^[\u4e00-\u9fa5_a-zA-Z0-9]+$/, message: '请输入中英文/数字/下划线', trigger: 'blur' },
    {validator: validateModelname, trigger: 'blur'}
  ]
}

// 保存模型时模型名不能为'无'
function validateModelname(rule, value, callback) {
  if (value === '无') {
    callback(new Error('模型名不能为“无”'));
  } else {
    callback();
  }
}

// 在使用教程中滚动到指定部分的方法  
const scrollTo = (sectionId: any) => {
  if (userHelpDialogScrollbar.value) {
    const element = userHelpDialogScrollbar.value.querySelector(`#${sectionId}`);
    if (element) {
      userHelpDialogScrollbar.value.scrollTop = element.offsetTop;
    }
  }
};

const router = useRouter()  // 页面路由

const dialogVisible = ref(false)

const activeName = ref('first')    // 控制标签页

const modelsDrawer = ref(false)   // 控制模型列表的抽屉
const dataDrawer = ref(false)     // 控制数据文件的抽屉

//控制按钮失效变量
const canStartProcess = ref(true)

const canCompleteModeling = computed(() => {
  if (nodeList.value.length > 0 && !modelHasBeenSaved) {
    return false
  } else {
    return true
  }
})
const canCheckModel = ref(true)
const canSaveModel = ref(true)
const canShutdown = ref(true)

// menuList2是为了显示算法列表，以及完成算法参数定义等操作，所定义的数据结构
// 其中节点的id为算法的id，label为算法的名称，parameters为算法的参数，use_algorithm为当前该模块所使用的算法名称，tip_show为是否显示提示信息的标志，tip为提示信息
const menuList2 = ref([{
  label: '数据预处理', id: '1', options: [
    {
      label: '插值处理', id: '1.1', use_algorithm: null, parameters: {
        'neighboring_values_interpolation': {},
        'bicubic_interpolation': {},
        'lagrange_interpolation': {},
        'newton_interpolation': {},
        'linear_interpolation': {},
        'deeplearning_interpolation': {},
      }, tip_show: false, tip: '对输入信号进行插值', optional: false
    },
    {
      label: '特征提取', id: '1.2', use_algorithm: null, parameters: {
        'time_domain_features': {
          均值: false,
          方差: false,
          标准差: false,
          偏度: false,
          峰度: false,
          四阶累积量: false,
          六阶累积量: false,
          最大值: false,
          最小值: false,
          中位数: false,
          峰峰值: false,
          整流平均值: false,
          均方根: false,
          方根幅值: false,
          波形因子: false,
          峰值因子: false,
          脉冲因子: false,
          裕度因子: false
        },
        'frequency_domain_features': {
          重心频率: false,
          均方频率: false,
          均方根频率: false,
          频率方差: false,
          频率标准差: false,
          谱峭度的均值: false,
          谱峭度的峰度: false,
        },
        'time_frequency_domain_features': {
          均值: false,
          方差: false,
          标准差: false,
          峰度: false,
          偏度: false,
          四阶累积量: false,
          六阶累积量: false,
          最大值: false,
          最小值: false,
          中位数: false,
          峰峰值: false,
          整流平均值: false,
          均方根: false,
          方根幅值: false,
          波形因子: false,
          峰值因子: false,
          脉冲因子: false,
          裕度因子: false,
          重心频率: false,
          均方频率: false,
          均方根频率: false,
          频率方差: false,
          频率标准差: false,
          谱峭度的均值: false,
          谱峭度的峰度: false,
        },
        'time_domain_features_multiple': {
          均值: false,
          方差: false,
          标准差: false,
          偏度: false,
          峰度: false,
          四阶累积量: false,
          六阶累积量: false,
          最大值: false,
          最小值: false,
          中位数: false,
          峰峰值: false,
          整流平均值: false,
          均方根: false,
          方根幅值: false,
          波形因子: false,
          峰值因子: false,
          脉冲因子: false,
          裕度因子: false
        },
        'frequency_domain_features_multiple': {
          重心频率: false,
          均方频率: false,
          均方根频率: false,
          频率方差: false,
          频率标准差: false,
          谱峭度的均值: false,
          谱峭度的峰度: false,
        },
        'time_frequency_domain_features_multiple': {
          均值: false,
          方差: false,
          标准差: false,
          峰度: false,
          偏度: false,
          四阶累积量: false,
          六阶累积量: false,
          最大值: false,
          最小值: false,
          中位数: false,
          峰峰值: false,
          整流平均值: false,
          均方根: false,
          方根幅值: false,
          波形因子: false,
          峰值因子: false,
          脉冲因子: false,
          裕度因子: false,
          重心频率: false,
          均方频率: false,
          均方根频率: false,
          频率方差: false,
          频率标准差: false,
          谱峭度的均值: false,
          谱峭度的峰度: false
        },
      }, tip_show: false, tip: '手工提取输入信号的特征', optional: true
    },
    {
      label: '无量纲化', id: '1.5', use_algorithm: null, parameters: {
        'max_min': {useLog: false},
        'z-score': {useLog: false},
        'robust_scaler': {useLog: false},
        'max_abs_scaler': {useLog: false},
      }, tip_show: false, tip: '对输入数据进行无量纲化处理', optional: true
    },
    {
      label: '特征选择', id: '1.3', use_algorithm: null, parameters: {
        'feature_imp': {rule: 1, threshold1: 0.1, threshold2: 0.1},
        'mutual_information_importance': {rule: 1, threshold1: 0.1, threshold2: 0.1},
        'correlation_coefficient_importance': {rule: 1, threshold1: 0.1, threshold2: 0.1},
        'feature_imp_multiple': {rule: 1, threshold1: 0.1, threshold2: 0.1},
        'mutual_information_importance_multiple': {rule: 1, threshold1: 0.1, threshold2: 0.1},
        'correlation_coefficient_importance_multiple': {rule: 1, threshold1: 0.1, threshold2: 0.1},
      }, tip_show: false, tip: '对提取到的特征进行特征选择', optional: true
    },
    {
      label: '小波变换', id: '1.4', use_algorithm: null, parameters: {
        'wavelet_trans_denoise': {'wavelet': '', 'wavelet_level': ''},
      }, tip_show: false, tip: '对输入信号进行小波变换', optional: true
    }
  ], tip_show: false, tip: '包含添加噪声、插值以及特征提取等'
},
{
  label: '故障检测', id: '2', options: [
    {
      label: '故障诊断', id: '2.1', use_algorithm: null, parameters: {
        'random_forest': {},
        'svc': {},
        'gru': {},
        'lstm': {},
        'random_forest_multiple': {},
        'svc_multiple': {},
        'gru_multiple': {},
        'lstm_multiple': {},
        'ulcnn': {},
        'ulcnn_multiple': {},
        'spectrumModel': {},
        'spectrumModel_multiple': {},
        // 'private_fault_diagnosis_deeplearning': '',
      }, tip_show: false, tip: '根据提取特征对输入信号作故障诊断', optional: false
    },
    {
      label: '故障预测', id: '2.2', use_algorithm: null, parameters: {
        'linear_regression': {},
        'linear_regression_multiple': {},
      }, tip_show: false, tip: '根据提取的信号特征对输入信号进行故障预测', optional: false
    }]
},
{
  label: '健康评估', id: '3', options: [
    {
      label: '层次分析模糊综合评估', id: '3.1', use_algorithm: null, parameters: {
        'FAHP': {},
        'FAHP_multiple': {},
      }, tip_show: false, tip: '将模糊综合评价法和层次分析法相结合的评价方法', optional: false
    },
    {
      label: '层次朴素贝叶斯评估', id: '3.2', use_algorithm: null, parameters: {
        'BHM': {},
        'BHM_multiple': {},
      }, tip_show: false, tip: '使用朴素贝叶斯方法的评价方法', optional: false
    },
    {
      label: '层次逻辑回归评估', id: '3.3', use_algorithm: null, parameters: {
        'AHP': {},
        'AHP_multiple': {},
      }, tip_show: false, tip: '使用层次逻辑回归方法的评价方法', optional: false
    },
  ]
},
  // {
  //   label: '增值组件', id: '-1', options: [
  //     {
  //       label: '插值处理', id: '1.1', use_algorithm: null, parameters: {
  //         'private_interpolation': '',
  //       }, tip_show: false, tip: '使用专有插值处理方法', optional: false
  //     } ,
  //     {label: '特征提取', id: '1.2', use_algorithm: null, parameters: {
  //       'private_feature_extraction':'',
  //       }},
  //     {
  //       label: '无量纲化', id: '1.5', use_algorithm: null, parameters: {
  //         'private_scaler': {useLog: false, algorithm: ''}
  //       }, tip_show: false, tip: '使用专有无量纲化处理方法', optional: true
  //     },
  //     {
  //       label: '特征选择', id: '1.3', use_algorithm: null, parameters: {
  //         'extra_feature_selection': {rule: 1, threshold1: 0.1, threshold2: 0.1}
  //       }, tip_show: false, tip: '使用专有特征选择方法', optional: true
  //     },
  //     {
  //       label: '小波变换', id: '1.4', use_algorithm: null, parameters: {
  //         'extra_wavelet_transform': ''
  //       }, tip_show: false, tip: '对输入信号进行小波变换', optional: true
  //     },
  //     {
  //       label: '故障诊断', id: '2.1', use_algorithm: null, parameters: {

  //         'private_fault_diagnosis_deeplearning': '',
  //         'private_fault_diagnosis_machine_learning': '',
  //       }, tip_show: false, tip: '使用专有故障诊断方法', optional: false
  //     },
  //     {
  //       label: '故障预测', id: '2.2', use_algorithm: null, parameters: {

  //         'private_fault_prediction': {}
  //       }, tip_show: false, tip: '使用专有故障预测方法', optional: false
  //     },
  //     {
  //       label: '健康评估', id: '3.4', use_algorithm: null, parameters: {
  //         'extra_health_evaluation': ''
  //       }, tip_show: false, tip: '使用专有健康评估的评价方法', optional: false
  //     },

  //   ],
  // },

]);


// 该方法用于判断是否显示可视化建模区的背景图片
const background_IMG = () => {
  if (nodeList.value.length == 0) {
    document.querySelector('.el-main')?.classList.add('has-background');

  }
  if (nodeList.value.length >= 1) {
    document.querySelector('.el-main')?.classList.remove('has-background');
    document.querySelector('.el-main').style.backgroundImage = ''

  }
}

// 算法参数的推荐值，目前包括小波变换的变换类型和变换层数、特征选择的依据规则及相应阈值
const recommendParams = {
  'wavelet': [{value: 'db1', label: 'db1'}, {value: 'db2', label: 'db2'}, {
    value: 'sym1',
    label: 'sym1'
  }, {value: 'sym2', label: 'sym2'}, {value: 'coif1', label: 'coif1'}],
  'wavelet_level': [{value: 1, label: '1'}, {value: 2, label: '2'}, {value: 3, label: '3'}],
  // 规则一的各算法的阈值推荐值
  'threshold1': {
    'feature_imp': [{value: 0.005, label: '0.005'}, {value: 0.01, label: '0.01'}, {
      value: 0.02,
      label: '0.02'
    }, {value: 0.03, label: '0.03'}, {value: 0.04, label: '0.04'}, {value: 0.05, label: 0.05}, {
      value: 0.1,
      label: 0.1
    }],
    'mutual_information_importance': [{value: 0.1, label: '0.1'}, {value: 0.2, label: '0.2'}, {
      value: 0.3,
      label: '0.3'
    }, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'},],
    'mutual_information_importance_multiple': [{value: 0.3, label: '0.3'}, {value: 0.35, label: '0.35'}, {
      value: 0.4,
      label: '0.4'
    }, {value: 0.45, label: '0.45'}, {value: 0.5, label: '0.5'}],
    'feature_imp_multiple': [{value: 0.01, label: '0.01'}, {value: 0.03, label: '0.03'}, {
      value: 0.05,
      label: '0.05'
    }, {value: 0.06, label: '0.06'}],
    'correlation_coefficient_importance_multiple': [{value: 0.58, label: '0.58'}, {
      value: 0.6,
      label: '0.6'
    }, {value: 0.62, label: '0.62'}, {value: 0.64, label: '0.64'}],
    'correlation_coefficient_importance': [{value: 0.1, label: '0.1'}, {value: 0.2, label: '0.2'}, {
      value: 0.3,
      label: '0.3'
    }, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'},]
  },

  // 规则二的各算法的阈值推荐值
  'threshold2': {
    'feature_imp': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {
      value: 0.6,
      label: '0.6'
    }, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}],
    'mutual_information_importance': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {
      value: 0.5,
      label: '0.5'
    }, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}],
    'feature_imp_multiple': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {
      value: 0.5,
      label: '0.5'
    }, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}],
    'mutual_information_importance_multiple': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {
      value: 0.5,
      label: '0.5'
    }, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}],
    'correlation_coefficient_importance_multiple': [{value: 0.2, label: '0.2'}, {value: 0.3, label: '0.3'}, {
      value: 0.4,
      label: '0.4'
    }, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {
      value: 0.8,
      label: '0.8'
    }, {value: 1, label: 1}],
    'correlation_coefficient_importance': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {
      value: 0.5,
      label: '0.5'
    }, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}]
  },
  // 'thresholdImpSingle1': [{value: 0.005, label: '0.005'}, {value: 0.01, label: '0.01'}, {value: 0.02, label: '0.02'}, {value: 0.03, label: '0.03'}, {value: 0.04, label: '0.04'}, {value: 0.05, label: 0.05}, {value: 0.1, label: 0.1}],
  // 'thresholdImgSingle2': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}]
  'scaleUseLogs': [{value: true, label: '使用训练模型时对数据的标准化方法'}, {
    value: false,
    label: '不使用训练模型时对数据的标准化方法'
  }]
}


// 监听特征选择的规则对应的阈值的初始值，以适应性的调整阈值的初始值

// 用于显示算法介绍
const introductionToShow = ref('# 你好世界')  // 需要展示在可视化建模区的算法介绍
const showPlainIntroduction = ref(false)

// 点击标签页切换单传感器和多传感器算法
// const handleClick = (tab, event) => {
//   console.log(tab, event)
// }

// 算法介绍，点击算法选择区内的具体算法，将其算法介绍展示在可视化建模区
const showIntroduction = (algorithm: string) => {
  resultsViewClear()
  showStatusMessage.value = false
  showPlainIntroduction.value = true
  introductionToShow.value = plainIntroduction[algorithm]

}


// 算法选择菜单下拉展示
const menuDetailsSecond = ref({})

const menuDetailsThird = ref({})


// 点击二级目录进行展开时，在结果可视化区域显示对应二级目录功能下所有算法的优劣比较
const contrastVisible = ref(false);

// 当点击二级目录时，展示对应三级目录下所有算法的优劣，此处为不同模块下各个算法优劣比较的表格的markdown代码

const contrastToShow = ref('')

const clickAtSecondMenu = (option: any) => {
  // 当点击二级目录时，展开对应二级目录下的三级目录
  menuDetailsThird.value[option.label] = !menuDetailsThird.value[option.label]

  // 在结果可视化区域显示对应二级目录功能下所有算法的优劣比较
  let secondAlgorithm = option.label
  if (secondAlgorithm === '层次分析模糊综合评估' || secondAlgorithm === '层次朴素贝叶斯评估' || secondAlgorithm === '层次逻辑回归评估' || secondAlgorithm === '健康评估') {
    secondAlgorithm = '层次分析健康评估'
  }
  contrastToShow.value = contrastOfAlgorithm[secondAlgorithm]

  resultsViewClear()
  contrastVisible.value = true
}

// 特征提取所选择的特征
// const features = ref(['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
//   '波形因子', '峰值因子', '脉冲因子', '裕度因子', '重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的峰度'])

const features = ref([])

//双向链表用于存储调用的模块顺序
class ListNode {
  constructor(value) {
    this.value = value;
    this.next = null;
  }
}

class LinkedList {
  constructor() {
    this.head = null;
    this.tail = null;
  }

  // 添加新元素到链表尾部
  append(value) {
    const newNode = new ListNode(value);

    if (!this.head) {
      this.head = newNode;
      this.tail = newNode;
    } else {
      this.tail.next = newNode;
      this.tail = newNode;
    }
  }

  // 在链表的头部添加新节点
  insertAtHead(value) {
    const newNode = new ListNode(value);

    if (!this.head) {
      this.head = newNode;
      this.tail = newNode;
    } else {
      newNode.next = this.head;
      this.head = newNode;
    }
  }

  // 打印链表所有元素
  print() {
    let current = this.head;
    while (current) {
      console.log(current.value);
      current = current.next;
    }
  }

  get_all_nodes() {
    let current = this.head;
    let nodeList = []
    while (current) {
      nodeList.push(current.value)
      current = current.next;
    }
    return nodeList
  }

  length() {
    if (this.head) {
      let len = 1
      let p = this.head.next
      while (p) {
        p = p.next
        len += 1
      }
      return len
    }
    return 0
  }

  search(target_value) {
    if (this.head == null) {
      return false
    } else {
      let current = this.head
      while (current) {
        if (current.value == target_value) {
          return current
        }
        current = current.next
      }
      return false
    }
  }

  searchPre(targetValue) {
    if (this.head == null) {
      return false
    } else {
      let current = this.head
      while (current && current.next) {
        if (current.next.value == targetValue) {
          return current
        }
        current = current.next
      }
      return false
    }
  }
}

const logout = () => {
  router.push('/')
}

// 标签与节点id的转换
const displayLabelToId = (displayLabel) => {
  nodeList.value.forEach(node => {
    if (node.display_label == displayLabel) {
      return node.id
    }
  })
}

// 节点标签到节点标识id的转换
function labelToId(label) {
  let nodeList1 = nodeList.value.slice()
  let nodeIdToFind = 0
  nodeList1.forEach(node => {
    if (node.label == label) {
      nodeIdToFind = node.id
    }
  })
  return nodeIdToFind
}

const userRole = ref('');  // 用户登录时所选择的角色，用于区分超级用户和普通用户的不同界面

// 将建立模型的连线操作、用户名设置、区分普通用户和系统用户的功能, 挂在到onMounted中
const linkedList = new LinkedList()
onMounted(() => {
  username.value = window.localStorage.getItem('username') || '用户名未设置'
  userRole.value = window.localStorage.getItem('role') || '无效的用户'
  // console.log('username: ', username.value)
  console.log('userRole: ', userRole.value)
  //获取用户上传的增值组件并构造对应的目录结构体
  getExtraAlgorithmMao()
  // 当进行建模的时候隐藏可视化建模区的背景文字
  document.querySelector('.el-main').classList.add('has-background');
  plumbIns = jsPlumb.getInstance()
  jsPlumbInit()

  plumbIns.bind("connection", function (info) {
    let sourceId = info.connection.sourceId
    let targetId = info.connection.targetId

    let id_to_label = {}

    nodeList.value.forEach(node => {
      let id = node.id
      let label = node.label
      id_to_label[id] = label
    })
    if (linkedList.head == null) {
      linkedList.append(id_to_label[sourceId])
      linkedList.append(id_to_label[targetId])
    } else {
      if (linkedList.head.value == id_to_label[targetId]) {
        linkedList.insertAtHead(id_to_label[sourceId])
      } else {
        linkedList.append(id_to_label[targetId])
      }
    }
    // 除去在linkedList中的节点，其他节点不能作为连线操作的出发点
    let linked = linkedList.get_all_nodes()
    // for(let [value, key] of id_to_label){
    //   if (linked.indexOf(key) == -1){
    //     plumbIns
    //   }
    // }
    // console.log('linked: ' + linked)
  })
})

const deff = {
  jsplumbSetting: {
    // 动态锚点、位置自适应
    Anchors: ['Right', 'Left'],
    anchor: ['Right', 'Left'],
    // 容器ID
    Container: 'efContainer',
    // 连线的样式，直线或者曲线等，可选值:  StateMachine、Flowchart，Bezier、Straight
    // Connector: ['Bezier', {curviness: 100}],
    // Connector: ['Straight', { stub: 20, gap: 1 }],
    Connector: ['Flowchart', {stub: 30, gap: 1, alwaysRespectStubs: false, midpoint: 0.5, cornerRadius: 10}],
    // Connector: ['StateMachine', {margin: 5, curviness: 10, proximityLimit: 80}],
    // 鼠标不能拖动删除线
    ConnectionsDetachable: false,
    // 删除线的时候节点不删除
    DeleteEndpointsOnDetach: false,
    /**
     * 连线的两端端点类型：圆形
     * radius: 圆的半径，越大圆越大
     */
    Endpoint: ['Dot', {radius: 8, cssClass: 'ef-dot', hoverClass: 'ef-dot-hover'}],

    EndpointStyle: {fill: '#808080', outlineWidth: 3,},
    // 是否打开jsPlumb的内部日志记录
    LogEnabled: true,
    /**
     * 连线的样式
     */
    PaintStyle: {
      // 线的颜色
      stroke: '#808080',
      // 线的粗细，值越大线越粗
      strokeWidth: 7,
      // 设置外边线的颜色，默认设置透明
      outlineStroke: 'transparent',
      // 线外边的宽，值越大，线的点击范围越大
      outlineWidth: 5,
    },
    DragOptions: {cursor: 'pointer', zIndex: 2000},
    ConnectionOverlays: [
      ['Custom', {
        create() {
          const el = document.createElement('div')
          // el.innerHTML = '<select id=\'myDropDown\'><option value=\'foo\'>foo</option><option value=\'bar\'>bar</option></select>'
          return el
        },
        location: 0.7,
        id: 'customOverlay',
      }],
    ],

    Overlays: [
      // 箭头叠加
      ['Arrow', {
        width: 25, // 箭头尾部的宽度
        length: 8, // 从箭头的尾部到头部的距离
        location: 1, // 位置，建议使用0～1之间
        direction: 1, // 方向，默认值为1（表示向前），可选-1（表示向后）
        foldback: 0.623, // 折回，也就是尾翼的角度，默认0.623，当为1时，为正三角
      }],

      ['Label', {label: '', location: 0.1, cssClass: 'aLabel',}],

    ],
    // 绘制图的模式 svg、canvas
    RenderMode: 'canvas',
    // 鼠标滑过线的样式
    HoverPaintStyle: {stroke: '#808080', strokeWidth: 10},
    // 滑过锚点效果
    EndpointHoverStyle: {fill: '#808080'},
    Scope: 'jsPlumb_DefaultScope', // 范围，具有相同scope的点才可连接
  },
  /**
   * 连线参数
   */
  jsplumbConnectOptions: {
    isSource: true,
    isTarget: true,
    // 动态锚点、提供了4个方向 Continuous、AutoDefault
    // anchor: 'Continuous',
    // anchor: ['Continuous', { faces: ['left', 'right'] }],
    // 设置连线上面的label样式
    labelStyle: {
      cssClass: 'flowLabel',
    },
  },
  /**
   * 源点配置参数
   */
  jsplumbSourceOptions: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: '.node-drag',
    filterExclude: false,
    anchor: ['Continuous', {faces: ['right']}],
    // 是否允许自己连接自己
    allowLoopback: false,
    maxConnections: -1,
  },

  jsplumbSourceOptions2: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: '.node-drag',
    filterExclude: false,
    // anchor: 'Continuous',
    // 是否允许自己连接自己
    allowLoopback: true,
    connector: ['Flowchart', {curviness: 50}],
    connectorStyle: {
      // 线的颜色
      stroke: 'red',
      // 线的粗细，值越大线越粗
      strokeWidth: 1,
      // 设置外边线的颜色，默认设置透明
      outlineStroke: 'transparent',
      // 线外边的宽，值越大，线的点击范围越大
      outlineWidth: 10,
    },
    connectorHoverStyle: {stroke: 'red', strokeWidth: 2},
  },
  jsplumbTargetOptions: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: '.node-drag',
    filterExclude: false,
    // 是否允许自己连接自己
    anchor: ['Continuous', {faces: ['left']}],
    allowLoopback: false,
    dropOptions: {hoverClass: 'ef-drop-hover'},
  },
}


const done = ref(false)      // 控制模型是否可以拖拽，当其值为true时不可拖拽模型
const dialogModle = ref(false)  // 保存模型对话框框

let modelCheckRight = false  // 为真时表明通过模型检查

// 检查模型
const checkModel = () => {
  console.log("进入检查模型", linkedList.get_all_nodes())
  let idToModule = {}
  let algorithms = []
  let algorithmSchedule = []
  let moduleSchedule = []
  // 如果只有一个模块则不需要建立流程
  if (nodeList.value.length == 1) {
    moduleSchedule.push(nodeList.value[0].label)  // 模块名称拼接的字符串
    algorithmSchedule.push(nodeList.value[0].label_display)  // 算法名称拼接的字符串
    // 根据是否包含多传感器的算法，判断是否是针对多传感器的模型
    if (algorithmSchedule.indexOf("多传感器") != -1) {
      contentJson.multipleSensor = true  // 默认为false，即对单传感器数据的模型
    }
  } else {

    // 如果有多个模块则需要根据用户的连接动作去形成正确的模型流程
    let allConnections = plumbIns.getConnections();
    console.log('all_connections: ', allConnections)
    // 获取连线元素的单向映射
    let connectionsMap: any = {};
    allConnections.forEach(connection => {
      const sourceId = connection.sourceId;
      const targetId = connection.targetId;

      // 如果源元素ID不在connectionsMap中，则初始化为空数组  
      if (!connectionsMap[sourceId]) {
        connectionsMap[sourceId] = [];
      }
      connectionsMap[sourceId].push(targetId);
    })

    // 寻找用户建立的模型流程逻辑上的第一个元素
    function findStartElement(connectionsMap: any) {
      // 创建一个集合来存储所有元素的ID  
      const allElements = new Set(Object.keys(connectionsMap).concat(...Object.values(connectionsMap).map(list => list)));

      // 遍历所有元素，查找没有入度的元素  
      for (const elementId of allElements) {
        let hasIncomingConnection = false;
        for (const connections of Object.values(connectionsMap)) {
          if (connections.includes(elementId)) {
            hasIncomingConnection = true;
            break;
          }
        }
        if (!hasIncomingConnection) {
          return elementId;     // 找到没有入度的元素，即起点
        }
      }

      // 如果没有找到没有入度的元素，则可能图不是线性的，或者connectionsMap构建有误  
      throw new Error("未找到起点元素.");
    }

    let startElementId = findStartElement(connectionsMap);

    // 寻找逻辑上的下一个元素
    function findNextElementIdInSequence(currentElementId, connectionsMap) {

      const connections = connectionsMap[currentElementId];

      // 建立模型时，模型序列是线性的 
      if (connections && connections.length > 0) {
        return connections[0]; // 返回序列中的下一个元素ID
      }
      return null;
    }

    // 生成所建立模型的运行流程
    function traverseLinearSequence(startElementId, connectionsMap, visited = new Set(), order = []) {
      // 检查是否已访问过当前元素  
      if (visited.has(startElementId)) {
        return;
      }

      visited.add(startElementId); // 标记为已访问  
      order.push(startElementId); // 将元素添加到顺序数组中  

      let nextElementId = findNextElementIdInSequence(startElementId, connectionsMap);

      if (nextElementId !== null) {
        // 递归遍历下一个元素
        traverseLinearSequence(nextElementId, connectionsMap, visited, order);
      }
      // 最终返回模型的运行流程
      return order;
    }

    let sequenceOrder = traverseLinearSequence(startElementId, connectionsMap);
    console.log('sequenceOrder: ', sequenceOrder);
    nodeList.value.forEach(node => {
      let id = node.id
      let label = node.label
      let algorithm = node.label_display
      idToModule[id] = label
      algorithms.push(algorithm)
    })

    sequenceOrder.forEach(id => {
      moduleSchedule.push(idToModule[id])
    });

    // 形成表示具体算法模块连接顺序的字符串
    for (let i = 0; i < moduleSchedule.length; i++) {
      let module = moduleSchedule[i]
      nodeList.value.forEach(node => {
        if (node.label == module) {
          algorithmSchedule.push(node.label_display)
        }
      });
    }
  }


  let moduleStr = Object.values(moduleSchedule).join('')   // 所有模块的名称按顺序拼接起来的字符串
  let algorithmStr = Object.values(algorithmSchedule).join('')  // 所有模块中的算法名称按顺序拼接起来的字符串

  // 判断子串后是否有更多的文本
  const moreText = (text, substring) => {
    const position = text.indexOf(substring);
    if (position === -1) {
      return false;
    }
    const endPosition = position + substring.length;
    return endPosition < text.length;
  }

  const includeHealthEvaluation = (moduleStr: string) => {
    if (moduleStr.match('层次分析模糊综合评估')) {
      return '层次分析模糊综合评估';
    } else if (moduleStr.match('层次朴素贝叶斯评估')) {
      return '层次朴素贝叶斯评估';
    } else if (moduleStr.match('层次逻辑回归评估')) {
      return '层次逻辑回归评估';
    } else if (moduleStr.match('健康评估')) {
      return '健康评估';
    } else {
      return '';
    }
  }

  // 判断一个子串后是否有另一个子串，其中subStrs2为包含需要寻找的子串的数组
  const checkSubstrings = (str, subStr1, subStrs2) => {
    const index1 = str.indexOf(subStr1);
    if (index1 !== -1) {
      // 如果 subStr1 存在  
      for (const subStr2 of subStrs2) {
        const index2 = str.indexOf(subStr2, index1 + subStr1.length);
        if (index2 !== -1) {
          // 如果在 subStr1 之后找到了 subStr2 中的任何一个则返回true  
          return true;
        }
      }
    }
    return false;
  }
  if (nodeList.value.length) {
    // 首先判断模型中是否有数据源
    if (!moduleStr.match('数据源')) {
      ElMessage.error('模型中未包含数据源，请添加数据源模块')
      return
    } else {
      if (nodeList.value.length == 1) {
        ElMessage.error('模型中仅包含数据源，无法运行，请添加其他模块')
        return
      }
      if (moduleStr.indexOf('数据源') > 0) {
        ElMessage.error('数据源模块必须位于模型中第一个位置')
        return
      }
    }
    moduleStr = moduleStr.replace('数据源', '')
    algorithmStr = algorithmStr.replace('数据源', '')
    console.log('moduleStr: ', moduleStr)
    console.log('algorithmStr: ', algorithmStr)
    // 首先判断模型中是否存在除了数据源之外的1个以上的模块，如果模型中只有一个模块，判断其是否可以独立地运行而不需要其他模块的支持
    if (nodeList.value.length == 2) {
      if (!moduleStr.match('插值处理') && !moduleStr.match('特征提取') && !algorithmStr.match('GRU的故障诊断') && !algorithmStr.match('LSTM的故障诊断') && !algorithmStr.match('小波变换')
          && !algorithmStr.match('一维卷积深度学习模型的故障诊断') && !algorithmStr.match('基于时频图的深度学习模型的故障诊断') && !moduleStr.match('无量纲化') && !algorithmStr.match('深度学习故障诊断')) {
        let tip
        if (moduleStr.match('故障诊断')) {
          tip = '模型中包含故障诊断，建议在此之前进行特征提取和特征选择等操作'
        } else if (moduleStr.match('层次分析模糊综合评估')) {
          tip = '模型中包含层次分析模糊综合评估，建议在此之前进行特征提取和特征选择等操作'
        } else if (moduleStr.match('特征选择')) {
          tip = '模型中包含特征选择，建议在此之前进行特征提取等操作'
        } else if (moduleStr.match('故障预测')) {
          tip = '模型中包含故障预测，建议在此之前进行故障诊断'
        }

        ElMessage({
          message: '该算法无法单独使用，请结合相应的算法,' + tip,
          type: 'warning',
          showClose: true
        })

        return
      } else {
        // 无量纲化要检查是否使用模型训练师使用的标准化方法，对输入的原始信号无法使用模型训练时使用的标准化方法进行无量纲化
        if (moduleStr.match('无量纲化')) {
          let node
          for (let item of nodeList.value) {
            if (item.label == '无量纲化') {
              node = item
              break
            }
          }
          if (node?.parameters[node.use_algorithm]['useLog'] == true) {

            ElMessageBox.alert('如果要使用模型训练时使用的标准化方法进行无量纲化，请确保无量纲化模块之前对数据进行了特征提取，或者在参数设置中选择不使用模型训练时使用的标准化方法', '提示', {
              confirmButtonText: '确定',
              draggable: true,
              buttonSize: 'medium',
            })

            return
          }
        }
        // 如果模块可以单独运行，再进行模型中各个模块的参数设置的检查
        let checkParamsRight = checkModelParams()
        if (checkParamsRight) {
          ElMessage({
            showClose: true,
            message: '模型正常，可以保存并运行',
            type: 'success'
          })
          modelCheckRight = true
          updateStatus('模型建立并已通过模型检查')
        } else {
          ElMessage({
            showClose: true,
            message: '请确保所有具有参数的模块的参数设置正确',
            type: 'warning'
          })
          return
        }
      }
    } else {
      // 检查模型中是否存在未被连接的模块
      if (linkedList.length() != nodeList.value.length) {
        ElMessage({
          message: '请确保图中所有模块均已建立连接，且没有多余的模块',
          type: 'warning'
        })
        return
      } else {
        // 模型正常连接的情况下进行模型逻辑以及模型参数的检查
        if (algorithmStr.match('多传感器') && algorithmStr.match('单传感器')) {
          ElMessage({
            showClose: true,
            message: '多传感器和单传感器的算法不能混合使用！',
            type: 'warning'
          })
          return
        }
        // if (moduleStr.match('特征选择故障诊断') && !moduleStr.match('特征提取特征选择故障诊断') && !moduleStr.match('特征提取特征选择无量纲化故障诊断')
        //     && !moduleStr.match('特征提取无量纲化特征选择故障诊断') && !algorithmStr.match('深度学习模型的故障诊断') && !algorithmStr.match('GRU的故障诊断') && !algorithmStr.match('LSTM的故障诊断')) {
        //   ElMessage({
        //     showClose: true,
        //     message: '因模型中包含故障诊断，建议在特征选择之前包含特征提取',
        //     type: 'warning'
        //   })
        //   return
        // } 
        if (moduleStr.match('故障诊断')) {
          // 如果是深度学习模型的故障诊断
          if (algorithmStr.match('深度学习模型的故障诊断') || algorithmStr.match('GRU的故障诊断') || algorithmStr.match('LSTM的故障诊断')) {
            if (moduleStr.indexOf('故障诊断') > 0) {
              // 检查深度学习模型的故障诊断之前是否包含不必要的模块
              let preModuleText = moduleStr.substring(0, moduleStr.indexOf('故障诊断'))
              if (preModuleText.match('特征提取') || preModuleText.match('特征选择') || preModuleText.match('无量纲化') || preModuleText.match('故障预测')) {
                ElMessage({
                  message: '深度学习模型的故障诊断不需要人工提取特征，因此其之前不需要包含如特征提取、特征选择等不必要的模块！',
                  type: 'warning',
                  showClose: true
                })
                return
              }
            }
            // 如果使用深度学习模型的故障诊断之后有其他的模块
            if (moreText(moduleStr, '故障诊断')) {
              let nextModuleText = moduleStr.substring(moduleStr.indexOf('故障诊断'), moduleStr.length)  //故障诊断模块之后的其他模块名拼接的字符串
              if (nextModuleText.match('故障预测') || nextModuleText.match('层次分析模糊综合评估') || nextModuleText.match('层次朴素贝叶斯评估') || nextModuleText.match('层次逻辑回归评估')) {
                // 如果同时包含故障预测以及层次分析模糊综合评估
                let current: string
                if (nextModuleText.match('故障预测') && (nextModuleText.match('层次分析模糊综合评估') || nextModuleText.match('层次朴素贝叶斯评估') || nextModuleText.match('层次逻辑回归评估'))) {
                  // 获取健康评估模块所在的位置
                  let healthEvaluationIndex
                  if (nextModuleText.match('层次分析模糊综合评估')) {
                    healthEvaluationIndex = nextModuleText.indexOf('层次分析模糊综合评估')
                  } else if (nextModuleText.match('层次朴素贝叶斯评估')) {
                    healthEvaluationIndex = nextModuleText.indexOf('层次朴素贝叶斯评估')
                  } else {
                    healthEvaluationIndex = nextModuleText.indexOf('层次逻辑回归评估')
                  }
                  // 如果健康评估模块位置在深度学习的故障预测组件之后，需要进一步进行手工特征的提取
                  if (nextModuleText.indexOf('故障预测') > healthEvaluationIndex) {
                    ElMessage({
                      message: '注意故障预测应该在层次分析模糊综合评估之前运行',
                      type: 'warning',
                      showClose: true
                    })
                    return
                  } else {
                    current = '故障预测'
                  }
                }

                // 因为之前的深度学习模型的故障诊断无法为故障预测或是健康评估提供样本特征，因此需要进行特征提取和特征选择
                if (nextModuleText.indexOf('故障预测') == -1) {
                  current = includeHealthEvaluation(moduleStr)
                  if (!current) {
                    message.error('模型检查出错，请重新链接模型')
                    return
                  }
                } else {
                  current = '故障预测'
                }
                // if (nextModuleText.indexOf('层次分析模糊综合评估') == -1){
                //   current = '故障预测'
                // }
                // if (includeHealthEvaluation(nextModuleText) == ''){
                //   current = '故障预测'
                // }
                // if (nextModuleText.indexOf('故障预测') > nextModuleText.indexOf('层次分析模糊综合评估')){
                //   current = '层次分析模糊综合评估'
                // }else{
                //   current = '故障预测'
                // }
                let preModuleText = nextModuleText.substring(0, nextModuleText.indexOf(current))
                if (!preModuleText.match('特征提取特征选择') && !preModuleText.match('特征提取无量纲化特征选择') && !preModuleText.match('特征提取特征选择无量纲化')) {
                  ElMessage({
                    message: '建议在深度学习模型的故障诊断之后包含特征提取和特征选择模块',
                    type: 'warning',
                    showClose: true
                  })
                  return
                }

              }
            }
          } else {
            // 如果是传统机器学习的故障诊断
            let preModuleText = moduleStr.substring(0, moduleStr.indexOf('故障诊断'))
            // if (!preModuleText.match('特征提取') && !preModuleText.match('特征选择') && !preModuleText.match('特征提取无量纲化特征选择')){
            //   ElMessage({
            //     message: '因模型中包含机器学习的故障诊断，建议在故障诊断之前包含特征提取及特征选择',
            //     type: 'warning'
            //   })
            //   return
            // }
            // if (!preModuleText.match('特征提取特征选择') && !preModuleText.match('特征提取无量纲化特征选择') && !preModuleText.match('特征提取特征选择无量纲化')){

            //   ElMessage({
            //     message: '建议在特征提取之后进行特征选择',
            //     type: 'warning',
            //     showClose: true
            //   })
            //   return 
            // }
            // 如果机器学习的故障诊断之前既不包含特征提取，也不包含特征选择
            if (!preModuleText.match('特征提取') && !preModuleText.match('特征选择')) {
              ElMessage({
                message: '建议在故障诊断之前进行特征提取和特征选择',
                type: 'warning',
                showClose: true
              })
              let preModule = linkedList.searchPre('故障诊断') // 寻找故障诊断之前的节点，即不符合规则的节点

              // 红色标明报错连线
              let sourceId = labelToId(preModule.value)
              let targetId = labelToId('故障诊断')

              plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
                stroke: '#E53935',
                strokeWidth: 7,
                outlineStroke: 'transparent',
                outlineWidth: 5,

              });
              return
            } else {
              // 如果特征提取和特征选择同时存在
              if (preModuleText.match('特征提取') && preModuleText.match('特征选择')) {
                // 如果特征提取在特征选择之后，此时逻辑错误
                if (preModuleText.indexOf('特征提取') > preModuleText.indexOf('特征选择')) {
                  ElMessage({
                    message: '建议在特征提取之后进行特征选择',
                    type: 'warning',
                    showClose: true
                  })
                  return
                }
              } else {
                // 如果只包含特征选择
                if (preModuleText.match('特征选择')) {
                  ElMessage({
                    message: '建议在特征提取之后再进行特征选择',
                    type: 'warning',
                    showClose: true
                  })
                  let preModule = linkedList.searchPre('特征选择') // 寻找特征选择之前的节点，即不符合规则的节点

                  // 红色标明报错连线
                  let sourceId = labelToId(preModule.value)
                  let targetId = labelToId('特征选择')

                  plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
                    stroke: '#E53935',
                    strokeWidth: 7,
                    outlineStroke: 'transparent',
                    outlineWidth: 5,

                  });
                  return
                }
                // 如果只包含特征提取
                else if (preModuleText.match('特征提取')) {
                  ElMessage({
                    message: '因模型中包含机器学习的故障诊断，建议在特征提取之后进行特征选择',
                    type: 'warning',
                    showClose: true
                  })
                  let current = linkedList.search('特征提取') // 寻找特征选择之前的节点，即不符合规则的节点
                  let next = current.next
                  // 红色标明报错连线
                  let sourceId = labelToId(current.value)
                  let targetId = labelToId(next.value)

                  plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
                    stroke: '#E53935',
                    strokeWidth: 7,
                    outlineStroke: 'transparent',
                    outlineWidth: 5,

                  });
                  return
                }

              }
            }
          }
        }
        // else {
        //   // 如果是机器学习的故障诊断
        //   if (moduleStr.indexOf('故障诊断') > 0) {
        //     let preModuleText = moduleStr.substring(0, moduleStr.indexOf('故障诊断'))
        //     if (!preModuleText.match('特征提取') && !preModuleText.match('特征选择') && !preModuleText.match('特征提取无量纲化特征选择')){
        //       ElMessage({
        //         message: '因模型中包含故障诊断，建议在故障诊断之前包含特征提取及特征选择',
        //         type: 'warning'
        //       })
        //       return
        //     }
        //     if (!preModuleText.match('特征提取特征选择') && !preModuleText.match('特征提取无量纲化特征选择') && !preModuleText.match('特征提取特征选择无量纲化')){

        //       ElMessage({
        //         message: '建议在特征提取之后进行特征选择',
        //         type: 'warning',
        //         showClose: true
        //       })
        //       return
        //     }
        //   }
        // }
        if (moduleStr.match('特征提取故障诊断')) {
          let sourceId = labelToId('特征提取')
          let current = linkedList.search('特征提取')
          let next = current.next.value
          let targetId = labelToId(next)

          let connection = plumbIns.getConnections({source: sourceId, traget: targetId})
          console.log('connection: ', connection)

          plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
            stroke: '#E53935',
            strokeWidth: 7,
            outlineStroke: 'transparent',
            outlineWidth: 5,

          });
          ElMessage({
            showClose: true,
            message: '因模型中包含故障诊断，建议在特征提取之后包含特征选择',
            type: 'warning'
          })
          return
        }
        if (moduleStr.indexOf('小波变换') > 0) {
          // 小波变换只能针对信号序列，之前不能已经进行了特征提取
          let preModuleText = moduleStr.substring(0, moduleStr.indexOf('小波变换'))
          if (preModuleText.match('特征提取')) {
            ElMessage({
              showClose: true,
              message: '模型中对原始信号进行了特征提取，而小波变换只能针对信号序列',
              type: 'warning'
            })
            let preModule = linkedList.searchPre('小波变换') // 寻找健康评估之前的节点，即不符合规则的节点

            // 红色标明报错连线
            let sourceId = labelToId(preModule.value)
            let targetId = labelToId('小波变换')

            let connection = plumbIns.getConnections({source: sourceId, traget: targetId})
            console.log('connection: ', connection)

            plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
              stroke: '#E53935',
              strokeWidth: 7,
              outlineStroke: 'transparent',
              outlineWidth: 5,

            });
            return
          }
        }
        if (includeHealthEvaluation(moduleStr)) {

          let healthEvaluation = includeHealthEvaluation(moduleStr)  //所包含的健康评估的组件名
          if (!moduleStr.match('特征提取')) {
            ElMessage({
              showClose: true,
              message: '因模型中包含层次分析模糊综合评估，建议在此之前包含特征提取',
              type: 'warning'
            })
            let current = linkedList.searchPre(healthEvaluation) // 寻找健康评估之前的节点，即不符合规则的节点

            // 红色标明报错连线
            let sourceId = labelToId(current.value)
            let targetId = labelToId(healthEvaluation)

            let connection = plumbIns.getConnections({source: sourceId, traget: targetId})
            console.log('connection: ', connection)

            plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
              stroke: '#E53935',
              strokeWidth: 7,
              outlineStroke: 'transparent',
              outlineWidth: 5,
            });
            return
          }
          if (moreText(moduleStr, healthEvaluation)) {
            let sourceId = labelToId(healthEvaluation)
            let current = linkedList.search(healthEvaluation)
            let next = current.next.value
            let targetId = labelToId(next)

            plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
              stroke: '#E53935',
              strokeWidth: 7,
              outlineStroke: 'transparent',
              outlineWidth: 5,

            });
            ElMessage({
              showClose: true,
              message: '注意健康评估之后无法连接更多的模块',
              type: 'warning'
            })
            return
          }

        }
        if (moduleStr.match('特征选择')) {
          let preModuleText = moduleStr.substring(0, moduleStr.indexOf('特征选择'))
          if (!preModuleText.match('特征提取')) {
            ElMessage({
              showClose: true,
              message: '因模型中包含特征选择，建议在此之前包含特征提取',
              type: 'warning'
            })
            let preModule = linkedList.searchPre('特征选择')
            let sourceId = labelToId(preModule)
            let targetId = labelToId('特征选择')

            // 通过jsPlumb实例对象的select方法选择连线，并设置连线的样式
            plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
              stroke: '#E53935',
              strokeWidth: 7,
              outlineStroke: 'transparent',
              outlineWidth: 5,

            });

            return
          }
        }
        if (algorithmStr.match('深度学习模型的故障诊断') || algorithmStr.match('GRU的故障诊断') || algorithmStr.match('LSTM的故障诊断')) {
          // 如果使用深度学习的故障诊断之前有其他模块，则要进行限定

          if (moduleStr.indexOf('故障诊断') != 0) {
            let preText = moduleStr.substring(0, moduleStr.indexOf('故障诊断'))
            // 检查使用深度学习的故障诊断之前是否有特征提取等不合理的模块
            if (preText.match('特征提取') || preText.match('特征选择') || preText.match('无量纲化')) {
              ElMessage({
                showClose: true,
                message: '使用深度学习模型的故障诊断不需要进行特征提取或是特征选择，请删除相关模块！',
                type: 'warning'
              })
              return
            }

          }
        }
        // if (moduleStr.match('层次分析模糊综合评估') && (moduleStr.match('LSTM的故障诊断') || moduleStr.match('GRU的故障诊断'))) {
        //   ElMessage({
        //     showClose: true,
        //     message: '使用深度学习模型的故障诊断无法为健康评估提供有效的评估依据，建议使用机器学习的故障诊断配合健康评估！',
        //     type: 'warning'
        //   })
        //   return
        // }
        // 健康评估之后无法再连接其他模块
        // if (includeHealthEvaluation(moduleStr) && moreText(moduleStr, '层次分析模糊综合评估')) {
        //   let sourceId = labelToId('层次分析模糊综合评估')
        //   let current = linkedList.search('层次分析模糊综合评估')
        //   let next = current.next.value
        //   let targetId = labelToId(next)

        //   plumbIns.select({ source: sourceId, target: targetId }).setPaintStyle({
        //     stroke: '#E53935',
        //     strokeWidth: 7,
        //     outlineStroke: 'transparent',
        //     outlineWidth: 5,

        //   });
        //   ElMessage({
        //     showClose: true,
        //     message: '注意健康评估之后无法连接更多的模块',
        //     type: 'warning'
        //   })
        //   return
        // }
        if (algorithmStr.match('多传感器') && algorithmStr.match('单传感器')) {
          ElMessage({
            showClose: true,
            message: '针对单传感器的算法无法与针对多传感器的算法共用',
            type: 'warning'
          })
          return
        }
        if (moduleStr.match('无量纲化')) {
          let node
          for (let item of nodeList.value) {
            if (item.label.match('无量纲化')) {
              node = item
              break
            }
          }
          let useLog = node.parameters[node.use_algorithm]['useLog']  // 获取无量纲化模块的参数
          // 无量纲化处理前没有其他模块
          if (moduleStr.indexOf('无量纲化') == 0) {
            // 检查无量纲化参数设置是否合理
            if (useLog == true) {

              ElMessageBox.alert('如果要使用模型训练时使用的标准化方法进行无量纲化，请确保无量纲化模块之前对数据进行了特征提取，或者在参数设置中选择不使用模型训练时使用的标准化方法', '提示', {
                confirmButtonText: '确定',
                draggable: true,
                buttonSize: 'medium',
              })
              return
            }
          } else {

            // 检查无量纲化处理前的其他模块是否符合无量纲化的运行规则
            let preModule = moduleStr.substring(0, moduleStr.indexOf('无量纲化'))
            if (preModule.match('特征提取') && useLog == false) {
              ElMessageBox.alert(
                  '因为无量纲化模块之前已经进行了特征提取，请在无量纲化的参数设置中选择使用模型训练时使用的标准化方法进行无量纲化',
                  '提示',
                  {
                    confirmButtonText: '确定',
                    draggable: true,
                    buttonSize: 'medium',
                  }
              )
              return
            } else if (!preModule.match('特征提取') && useLog == true) {
              ElMessageBox.alert(
                  '无量纲化模块之前未进行特征提取，因此无法使用模型训练时使用的标准化方法进行无量纲化，请在无量纲化的参数设置中选择不使用模型训练时使用的标准化方法进行无量纲化',
                  '提示',
                  {
                    confirmButtonText: '确定',
                    draggable: true,
                    buttonSize: 'medium',
                  }
              )
            }
          }

        }
        // if ((algorithmStr.match('LSTM的故障诊断') || algorithmStr.match('GRU的故障诊断')) && (checkSubstrings(moduleStr, '故障诊断', ['故障预测', '健康评估']))){
        //   let nextModuleText = moduleStr.substring(moduleStr.indexOf('故障诊断'), moduleStr.length)
        //   if (!nextModuleText.match('特征提取') && !nextModuleText.match('特征选择')){
        //     ElMessage({
        //       message: '注意深度学习的算法并不能为线性回归的故障预测或是健康评估提供需要的特征。',
        //       type: 'warning',
        //       showClose: true
        //     })
        //     return
        //   }
        //   // ElMessage({
        //   //   message: '注意深度学习的算法并不能为线性回归的故障预测或是健康评估提供需要的特征。',
        //   //   type: 'warning',
        //   //   showClose: true
        //   // })

        //   let sourceId = labelToId('故障诊断')
        //   let current = linkedList.search('故障诊断')
        //   // 红色表明报错连线
        //   let next = current.next.value       // 寻找目标连线的源节点和目标节点
        //   let targetId = labelToId(next)

        //   // let connection = plumbIns.getConnections({ source: sourceId, traget: targetId })
        //   // console.log('connection: ', connection)

        //   // 通过jsPlumb实例对象的select方法选择连线，并设置连线的样式
        //   plumbIns.select({ source: sourceId, target: targetId }).setPaintStyle({
        //     stroke: '#E53935',
        //     strokeWidth: 7,
        //     outlineStroke: 'transparent',
        //     outlineWidth: 5,

        //   });
        //   return
        // }
        // 检查模型中是否使用了深度学习模型的故障诊断
        const useDeepLearningModule = (algorithmStr: string) => {
          return algorithmStr.match('LSTM的故障诊断') || algorithmStr.match('GRU的故障诊断') || algorithmStr.match('一维卷积深度学习模型的故障诊断') || algorithmStr.match('时频图深度学习模型的故障诊断')
        }
        // 规定插值处理只能是在模型中的开始位置
        if (moduleStr.match('插值处理')) {
          if (moduleStr.indexOf('插值处理') != 0) {
            ElMessage({
              showClose: true,
              message: '插值处理只能处在模型中的开始位置',
              type: 'warning'
            })
            let preModule = linkedList.searchPre('插值处理')
            let sourceId = labelToId(preModule)
            let targetId = labelToId('插值处理')

            // 通过jsPlumb实例对象的select方法选择连线，并设置连线的样式
            plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
              stroke: '#E53935',
              strokeWidth: 7,
              outlineStroke: 'transparent',
              outlineWidth: 5,

            });
            return
          }
        }
        if (moduleStr.match('故障预测')) {
          // 故障预测之前必须进行故障诊断
          if (moduleStr.indexOf('故障预测') <= 0) {
            ElMessage({
              showClose: true,
              message: '故障预测之前需要进行故障预测',
              type: 'warning'
            })

            return
          } else {
            let preModuleText = moduleStr.substring(0, moduleStr.indexOf('故障预测'))
            if (!preModuleText.match('故障诊断')) {
              ElMessage({
                showClose: true,
                message: '故障预测之前需要进行故障诊断',
                type: 'warning'
              })
              // 将报错的连线标注为红色

              let preModule = linkedList.searchPre('故障预测')
              let sourceId = labelToId(preModule.value)
              let targetId = labelToId('故障预测')

              // 通过jsPlumb实例对象的select方法选择连线，并设置连线的样式0
              plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
                stroke: '#E53935',
                strokeWidth: 7,
                outlineStroke: 'transparent',
                outlineWidth: 5,

              });
              return
            }

          }
        }
        if (moduleStr.match('故障诊断')) {
          let useDeepLearning = useDeepLearningModule(algorithmStr)
          if (moreText(moduleStr, '故障诊断')) {
            // 机器学习的故障诊断之后只能是进行故障预测或是健康评估

            if (!useDeepLearning && !checkSubstrings(moduleStr, '故障诊断', ['层次分析模糊综合评估', '故障预测', '层次朴素贝叶斯评估', '层次逻辑回归评估', '健康评估'])) {
              ElMessage({
                showClose: true,
                message: '注意故障诊断之后仅能进行故障预测或是健康评估！',
                type: 'warning'
              })
              // 将报错的连线标注为红色
              let sourceId = labelToId('故障诊断')
              let current = linkedList.search('故障诊断')
              let next = current.next.value       // 寻找目标连线的源节点和目标节点
              let targetId = labelToId(next)

              // 通过jsPlumb实例对象的select方法选择连线，并设置连线的样式
              plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
                stroke: '#E53935',
                strokeWidth: 7,
                outlineStroke: 'transparent',
                outlineWidth: 5,

              });

              return
            }
          }
          // 如果模型中包含SVM的故障诊断，则需要先加入无量纲化操作
          if (algorithmStr.match('SVM的故障诊断')) {
            if (!moduleStr.match('无量纲化') || !checkSubstrings(moduleStr, '无量纲化', ['故障诊断'])) {
              ElMessage({
                showClose: true,
                message: '因模型中包含SVM的故障诊断，需要在此之前加入z-score标准化操作',
                type: 'warning'
              })
              // 将报错的连线标注为红色
              let sourceId = labelToId('特征选择')
              let current = linkedList.search('特征选择')
              let next = current.next.value       // 寻找目标连线的源节点和目标节点
              let targetId = labelToId(next)

              // 通过jsPlumb实例对象的select方法选择连线，并设置连线的样式
              plumbIns.select({source: sourceId, target: targetId}).setPaintStyle({
                stroke: '#E53935',
                strokeWidth: 7,
                outlineStroke: 'transparent',
                outlineWidth: 5,

              });
              return
            } else {
              if (!algorithmStr.match('z-score标准化')) {
                ElMessage({
                  showClose: true,
                  message: '因模型中包含SVM的故障诊断，需要在此之前加入z-score标准化操作',
                  type: 'warning'
                })
                return
              }
            }

          }
        }
        // 进行模型参数设置的检查
        let check_params_right = checkModelParams()
        if (check_params_right) {
          ElMessage({
            showClose: true,
            message: '模型正常，可以保存并运行',
            type: 'success'
          })
          modelCheckRight = true
          updateStatus('模型建立并已通过模型检查')
        } else {
          ElMessage({
            showClose: true,
            message: '请确保所有具有参数的模块的参数设置正确',
            type: 'warning'
          })
          return
        }
      }
    }
  } else {
    ElMessage({
      message: '请先建立模型',
      type: 'warning'
    })
    return
  }
  canSaveModel.value = false
  // canStartProcess.value = false
}

// 进度条完成度
let processing = ref(false)
let percentage = ref(0)
// let timerId = null
let fastTimerId = null; // 快速定时器ID  
let slowTimerId = null; // 慢速定时器ID  

let responseResults = {}  // 从后端接收到的模型运行的结果数据

const username = ref('')  // 显示在用户界面中的用户名


// 拦截器拦截axios请求，如果请求被取消，则不发送请求  
// axios.interceptors.request.use(config => {  
//   // 如果请求被取消，则不发送请求  
//   if (config.cancelToken) {  
//     config.cancelToken = source.token;  
//   }  
//   return config;  
// });  
let cancel;

const source = axios.CancelToken.source();
cancel = source.cancel; // 暴露cancel函数  


//上传文件后，点击开始运行以运行程序
const startProgram = () => {

  if (!usingDatafile.value) {
    ElMessage({
      message: '请先加载数据',
      type: 'warning'
    })
    return
  }

  const data = new FormData()
  data.append("file_name", usingDatafile.value)  // 所使用的数据文件
  data.append('params', JSON.stringify(contentJson))  // 模型信息
  // console.log('params: ', contentJson)
  if (usingDatafile.value == '无') {
    ElMessage({
      message: '当前加载的数据为空，请先加载数据文件',
      type: 'error'
    })
    return
  }
  ElNotification.info({
    title: 'Waiting',
    message: '正在运行，请等待...'
  })
  canShutdown.value = false

  percentage.value = 0; // 重置进度条  

  fastTimerId = setInterval(() => {
    if (percentage.value < 50) {
      percentage.value += 10;
    } else {
      // 达到50%时，清除快速定时器并启动慢速定时器  
      clearInterval(fastTimerId);
      slowTimerId = setInterval(() => {
        if (percentage.value < 90) {
          percentage.value += 10;
        } else {
          // 达到100%时清除慢速定时器  
          clearInterval(slowTimerId);
        }
      }, 3000);
    }
  }, 1000);

  // 显示进度条
  resultsViewClear()
  processing.value = true
  showStatusMessage.value = false
  showPlainIntroduction.value = false

  api.post('user/run_with_datafile_on_cloud/', data,
      {
        headers: {"Content-Type": 'multipart/form-data'},
        cancelToken: source.token, // 将cancelToken传递给axios
      }
  ).then((response) => {
    if (response.data.code == 401) {
      ElMessageBox.alert('登录状态已失效，请重新登陆', '提示', {
        confirmButtonText: '确定',
        callback: (action: Action) => {
          router.push('/')
        },
      })
    }
    if (response.data.code === 200) {

      if (!processIsShutdown.value) {
        ElNotification.success({
          title: 'Success',
          message: '程序运行完成',
        })
        responseResults = response.data.results
        missionComplete.value = true
        statusMessageToShow.value = statusMessage.success

      } else {
        processIsShutdown.value = false
      }
    } else if (response.data.code == 404) {
      statusMessageToShow.value = statusMessage.error
      ElMessage({
        message: response.data.message,
        type: 'warning'
      })
    }
    clearInterval(fastTimerId);
    clearInterval(slowTimerId);
    setTimeout(function () {
      processing.value = false
    }, 700)
    percentage.value = 100;
    canShutdown.value = true

    resultsViewClear()

    showStatusMessage.value = true
    showPlainIntroduction.value = false
  })
      .catch(error => {

        if (error.response) {
          // 请求已发出，服务器响应了状态码，但不在2xx范围内
          console.log(error.response.status); // HTTP状态码
          console.log(error.response.statusText); // 状态消息

        } else if (error.request) {
          // 请求已发起，但没有收到响应
          console.log(error.request);
        } else {
          // 设置请求时触发了错误
          console.error('Error', error.message);
        }

        ElNotification.error({
          title: 'Error',
          message: '运行出错，' + error.response.data?.message,
        })
        loading.value = false
        processing.value = false

        canShutdown.value = true
        statusMessageToShow.value = statusMessage.error
        resultsViewClear()
        showStatusMessage.value = true
        missionComplete.value = false

      })
}

// 用于判断该程序是否是正常运行结束的，如果该变量为真，表示为手动终止运行
const processIsShutdown = ref(false)


// 终止模型的运行
const shutDown = () => {
  api.get('/shut').then((response: any) => {
    if (response.data.status == 'shutdown' && processing.value == true) {
      loading.value = false
      processing.value = false
      missionComplete.value = false
      ElNotification.info({
        title: 'INFO',
        message: '进程已终止'
      })
      resultsViewClear()
      processIsShutdown.value = true
      statusMessageToShow.value = statusMessage.shutdown
      showStatusMessage.value = true
      canShutdown.value = true
      // canStartProcess.value = false
      // cancel('Operation canceled by the user.');  
    }
  }).catch(function (error: any) {
    // 处理错误情况  
    ElNotification.error({
      title: 'ERROR',
      message: '请求终端进程失败'
    })
    console.log('请求中断进程失败：' + error)
  });
}


const isShow = ref(false)
const selects = ref(false)

const efContainerRef = ref()
const nodeRef = ref([])

const nodeList = ref([])   // 保存可视化建模区中的各节点的列表

// 前端向后端传递的要运行的模型的信息，由包括的模块、模块使用的算法、使用的参数、模块的运行顺序组成
const contentJson = {
  'modules': [],
  'algorithms': {},
  'parameters': {},
  'schedule': [],
  'multipleSensor': false  // 是否为多传感器数据
}

let plumbIns   // 实例化的jsPlumb对象，实现用户建模的连线操作
let missionComplete = ref(false)
let loading = ref(false)
let modelSetup = ref(false)

// 清除页面中的内容，包括使用的模型、文件和算法介绍等信息
const handleClear = () => {
  done.value = false
  nodeList.value = []  // 可视化建模区的节点列表
  // features.value = []  // 特征提取选择的特征
  features.value = ['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
    '波形因子', '峰值因子', '脉冲因子', '裕度因子', '重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的标准差', '谱峭度的峰度', '谱峭度的偏度']
  jsonClear()    // 向后端发送的模型信息
  isShow.value = false
  plumbIns.deleteEveryConnection()
  plumbIns.deleteEveryEndpoint()
  linkedList.head = null
  linkedList.tail = null
  missionComplete.value = false // 程序处理完成
  modelSetup.value = false   // 模型设置完成
  showPlainIntroduction.value = false
  showStatusMessage.value = false
  modelHasBeenSaved = false  //复用历史模型，不做模型检查
  toRectifyModel.value = false  // 禁用修改模型
  canCompleteModeling.value = true
  canCheckModel.value = true
  canSaveModel.value = true
  processIsShutdown.value = false
  canStartProcess.value = true   // 不可运行程序
  modelLoaded.value = '无'

  updateStatus('未建立模型')

  resultsViewClear()
}

// 用于清空向后端传递的要运行的模型的信息
const jsonClear = () => {
  contentJson.modules = []
  contentJson.algorithms = {}
  contentJson.parameters = {}
  contentJson.schedule = []
}
const jsPlumbInit = () => {
  plumbIns.importDefaults(deff.jsplumbSetting)
}

// 数据源节点的节点信息
const dataSourceNode = {id: '4', label: '数据源', parameters: {dataSource: {}}, optional: true}

// 自定义模块节点的节点信息
const customModuleNode = {id: '5', label: '自定义模块', parameters: {customModule: {}}, optional: true}

//处理拖拽，初始化节点的可连接状态
const handleDragend = (ev, algorithm, node) => {

  // 拖拽进来相对于地址栏偏移量
  const evClientX = ev.clientX
  const evClientY = ev.clientY
  let left
  if (evClientX < 300) {
    left = evClientX + 'px'
  } else {
    left = evClientX - 300 + 'px'
  }

  let top = 50 + 'px'
  const nodeId = node.id
  const nodeInfo = {
    label_display: labelsForAlgorithms[algorithm],   // 具体算法的名称
    label: node.label,      // 算法模块名称
    id: node.id,
    nodeId,
    nodeContainerStyle: {
      left: left,
      top: top,
    },
    use_algorithm: algorithm,
    parameters: node.parameters,
    optional: node.optional
  }

  // 针对时域或是频域特征给出不同的可选特征
  if (nodeInfo.label_display.indexOf('时域和频域') > -1) {
    features.value = ['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
      '波形因子', '峰值因子', '脉冲因子', '裕度因子', '重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的峰度']
  } else {
    if (nodeInfo.label_display.indexOf('时域特征') > -1) {
      features.value = ['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
        '波形因子', '峰值因子', '脉冲因子', '裕度因子']
    } else if (nodeInfo.label_display.indexOf('频域特征') > -1) {
      features.value = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的峰度']
    }
  }
  // console.log(nodeInfo)
  //算法模块不允许重复
  if (nodeList.value.length === 0) {
    nodeList.value.push(nodeInfo)
  } else {
    let isDuplicate = false;
    for (let i = 0; i < nodeList.value.length; i++) {
      let nod = nodeList.value[i];
      if (nod.id == node.id) {
        // window.alert('不允许出现重复模块');
        ElMessage({
          message: '不允许出现同一类别的算法',
          type: 'warning'
        })
        isDuplicate = true;
        break;
      }
    }
    // 向节点列表中添加新拖拽入可视化建模区中的模块
    if (!isDuplicate) {
      nodeList.value.push(nodeInfo);
    }
  }

  // 将节点初始化为可以连线的状态
  nextTick(() => {
    plumbIns.draggable(nodeId, {containment: "efContainer"})

    if (node.id < 4) {
      plumbIns.makeSource(nodeId, deff.jsplumbSourceOptions)
    }

    if (node.id == '4') {
      plumbIns.makeSource(nodeId, deff.jsplumbSourceOptions)
      return
    }

    plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)

  })
}

// 删除节点
const deleteNode = (nodeId) => {
  if (!modelSetup.value) {
    nodeList.value = nodeList.value.filter(node => node.id !== nodeId);
    plumbIns.deleteEveryConnection()
    plumbIns.deleteEveryEndpoint()
    linkedList.head = null
    linkedList.tail = null
    canCheckModel.value = true
    canStartProcess.value = true
    canShutdown.value = true
    canSaveModel.value = true
  }
}


// 处理可视化建模区中拖拽节点的操作
const handleMouseup = (ev, data) => { // 在图表中拖拽节点时，设置他的新的位置

  if (!done.value) {
    length = nodeList.value.length
    for (let i = 0; i < length; i++) {
      let node = nodeList.value[i]
      if (node.id === data.id) {
        // setTimeout(()=>{
        //   data.nodeContainerStyle.left = ev.clientX - 290 
        //   data.nodeContainerStyle.top = ev.clientY - 80 
        // }, 2)
        nodeList.value[i].nodeContainerStyle.left = ev.clientX - 355 + 'px'
        nodeList.value[i].nodeContainerStyle.top = ev.clientY - 105 + 'px'
      }
    }
  }

}

// const modelsetting = () => {
//   selects.value = !selects.value
// }

const dialogFormVisible = ref(false)  // 控制保存模型对话框的弹出，输入要保存的模型的名称

// 提交的模型相关信息
const modelInfoForm = ref({
  name: '',
})

// 检查模型参数设置
const checkModelParams = () => {
  for (let i = 0; i < nodeList.value.length; i++) {
    let dict = nodeList.value[i]
    console.log('dict.use_algorithm: ', dict.use_algorithm)
    if (!dict.use_algorithm) {
      return false
    } else {
      // 检查特征选择的规则参数
      if (dict.id == '1.3') {
        let threshold = false
        // 检查选择特征的规则参数是否正确设置
        let rule = dict.parameters[dict.use_algorithm].rule
        if (dict.parameters[dict.use_algorithm]['threshold' + rule]) {
          threshold = true
        }
        if (!threshold) {
          return false
        }
      } else if (dict.id == '1.2') {
        // 检查特征提取参数设置
        if (!features.value.length) {
          return false
        }
      } else {
        console.log('dict信息: ', dict)
        console.log('dict.use: ', dict.use_algorithm)
        console.log('dict.parameters: ', dict.parameters)
        // 检查一般算法模块的参数设置，参数设置不能为空
        let parameters = dict.parameters[dict.use_algorithm]
        console.log('parameters: ', parameters)
        if (!parameters) {
          console.log('parameters is null')
          return false
        } else {
          for (let key in parameters) {
            console.log("parameters", parameters[key])
            console.log("key....", key)
            if (parameters[key] === '' || parameters[key] === null) {
              console.log('parameters[key] is null: ', parameters[key])
              return false
            }
          }
        }
      }
    }
  }

  return true
}

//保存模型，并取消拖拽动作                 
const saveModelSetting = (saveModel, schedule) => {

  done.value = true

  // dialogFormVisible.value = true
  // selects.value = !selects.value
  jsonClear()
  for (let i = 0; i < nodeList.value.length; i++) {
    let dict = nodeList.value[i]

    if (!dict.use_algorithm) {
      ElMessage({
        message: '请设置每个算法的必选属性',
        type: 'error'
      })

      return
    }

    contentJson.algorithms[dict.label] = dict.use_algorithm
    if (!contentJson.modules.includes(dict.label) && dict.id !== '4') {
      contentJson.modules.push(dict.label);
    }

    // 选择特征提取需要提取的参数
    if (dict.id == '1.2') {
      let params = dict.parameters[dict.use_algorithm]
      if (!features.value.length) {
        ElMessage({
          message: '请设置每个算法的必选属性',
          type: 'error'
        })
        return
      }
      features.value.forEach(element => {
        if (params[element] == false) {
          params[element] = true
        }
      });
      contentJson.parameters[dict.use_algorithm] = params
      continue
    }
    contentJson.parameters[dict.use_algorithm] = dict.parameters[dict.use_algorithm]
    // console.log(dict.use_algorithm + '\'s params are: ' + dict.parameters[dict.use_algorithm])

  }
  if (!modelCheckRight && saveModel) {
    ElMessage({
      message: '请先建立模型并通过模型检查！',
      type: 'warning'
    })
    return
  }

  contentJson.schedule.length = 0

  if (!saveModel) {
    // 如果是加载的已保存的模型，则直接使用已保存模型的流程
    // console.log('schedule: ', schedule)
    contentJson.schedule = schedule
    // console.log('content_json: ', contentJson)
  } else {
    let current = linkedList.head;
    if (!current) {
      ElNotification({
        title: 'WARNING',
        message: '未建立流程，请先建立流程',
        type: 'warning',
      })
      return
    }
    // 使用自定义模型链表建立的流程
    while (current) {
      // if (current.value == '数据源') continue

      contentJson.schedule.push(current.value);
      current = current.next;
    }
  }

  dialogModle.value = saveModel
}

// 完成模型名称等信息的填写后，确定保存模型
const saveModelConfirm = () => {
  // 将模型信息保存到数据库
  let data = new FormData()
  data.append('model_name', modelInfoForm.value.name)
  let nodelistInfo = nodeList.value
  let modelInfo = {"nodeList": nodelistInfo, "connection": contentJson.schedule}
  data.append('model_info', JSON.stringify(modelInfo))

  api.post('/user/save_model/', data,
      {
        headers: {"Content-Type": 'multipart/form-data'}
      }
  ).then((response) => {
    if (response.data.code == 401) {
      ElMessageBox.alert('登录状态已失效，请重新登陆', '提示', {
        confirmButtonText: '确定',
        callback: (action: Action) => {
          router.push('/')
        },
      })
    }
    if (response.data.message == 'save model success') {
      ElMessage({
        message: '保存模型成功',
        type: 'success'
      })

      fetchModels()
      modelsDrawer.value = false       // 关闭历史模型抽屉
      dialogFormVisible.value = false    // 关闭提示窗口
      dialogModle.value = false
      canStartProcess.value = false     // 保存模型成功可以运行
      modelSetup.value = true                 // 模型保存完成
      modelLoaded.value = modelInfoForm.value.name  // 保存模型后，显示当前模型名称
      updateStatus('当前模型已保存')
    } else if (response.data.code == 400) {
      ElMessage({
        message: '已有同名模型，保存模型失败',
        type: 'error'
      })
    }
  }).catch(error => {
    ElMessage({
      message: '保存模型请求失败',
      type: 'error'
    })
    console.log('save model error: ', error)
  })
}

const show1 = ref(false)

// 结果可视化区域显示
const canShowResults = ref(false)

// 健康评估结果展示
const healthEvaluation = ref('')
const displayHealthEvaluation = ref(false)
const activeName1 = ref('first')
const healthEvaluationOfExample = ref('样本1')
const resultsBarOfAllExamples = ref<Object>({});
const levelIndicatorsOfAllExamples = ref({});
const statusOfExamples = ref({});
const suggestionOfAllExamples = ref({});
const finalSuggestion = ref('');
// const healthEvaluationFigure1 = ref('data:image/png;base64,')
// const healthEvaluationFigure2 = ref('data:image/png;base64,')
// const healthEvaluationFigure3 = ref('data:image/png;base64,')

const healthEvaluationDisplay = (results_object) => {

  // 各个样本健康评估的可视化结果
  displayHealthEvaluation.value = true

  let figure1 = results_object.层级有效指标_Base64
  let figure2 = results_object.二级指标权重柱状图_Base64
  let figure3 = results_object.评估结果柱状图_Base64
  let suggestions = results_object.评估建议

  resultsBarOfAllExamples.value = {}
  figure1.forEach((element: string, index: number) => {
    // console.log("element: ", element)
    // console.log("index: ", index)
    resultsBarOfAllExamples.value[`样本${index + 1}`] = 'data:image/png;base64,' + element
  });
  levelIndicatorsOfAllExamples.value = {}
  figure2.forEach((element: string, index: number) => {
    levelIndicatorsOfAllExamples.value[`样本${index + 1}`] = 'data:image/png;base64,' + element
  });
  statusOfExamples.value = {};
  figure3.forEach((element: string, index: number) => {
    statusOfExamples.value[`样本${index + 1}`] = 'data:image/png;base64,' + element
  });
  suggestionOfAllExamples.value = {};
  suggestions.forEach((element: string, index: number) => {
    suggestionOfAllExamples.value[`样本${index + 1}`] = element
  });

  // 最终评估结果
  finalSuggestion.value = results_object.最终评估结果
  let statusOfAllExamples: Object = results_object.各样本状态隶属度

  nextTick(() => {
    // 绘制故障样本与非故障样本数量饼状图
    var pieChartDom = document.getElementById('healthEvaluationPieChart');
    var pieChart = echarts.init(pieChartDom);
    var pieChartOption;

    pieChartOption = {
      title: {
        text: '评估结果中不同状态所属的样本数量比例',
        // subtext: 'Fake Data',
        left: 'center'
      },
      tooltip: {
        trigger: 'item'
      },
      legend: {
        orient: 'vertical',
        left: 'left'
      },
      series: [
        {
          name: '样本数量',
          type: 'pie',
          radius: '50%',
          data: [
            // { value: 580, name: 'Email' },
            // { value: 484, name: 'Union Ads' },
            // { value: 300, name: 'Video Ads' }
          ],
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
    };
    for (let [key, value] of Object.entries(statusOfAllExamples)) {
      pieChartOption.series[0].data.push({value: value, name: key})
    }
    console.log("pieChartOption.series[0].data: ", pieChartOption.series[0].data)
    pieChart.setOption(pieChartOption);
  })
  // healthEvaluation.value = results_object.评估建议
  // healthEvaluationFigure1.value = 'data:image/png;base64,' + figure1
  // healthEvaluationFigure2.value = 'data:image/png;base64,' + figure2
  // healthEvaluationFigure3.value = 'data:image/png;base64,' + figure3

}


// 特征提取结果展示
const displayFeatureExtraction = ref(false)
// const transformedData = ref([])
// const columns = ref([])
const numOfSensors = ref<number>(0)
const rawDataList = ref<Object[]>([])
const featuresSeriesList = ref<Object[]>([])
const featuresExtractionRawData = ref('传感器 1')

const featureExtractionDisplay = (resultsObject) => {


  // 获取后端传回的提取的特征
  let featuresWithName = Object.assign({}, resultsObject.features_with_name)
  let featuresName = featuresWithName.features_name
  let featuresToDrawLineChart = Object.assign({}, resultsObject.featuresToDrawLineChart)
  // let featuresGroupBySensor = Object.assign(featuresWithName.features_extracted_group_by_sensor)

  console.log("featuresToDrawLineChart: ", featuresToDrawLineChart)

  let num_frames = resultsObject.num_examples
  // 根据帧数num_frames生成x坐标的坐标轴
  let x_axis = []
  for (let i = 0; i < num_frames; i++) {
    x_axis.push('样本' + (i + 1) + `(${i * 2048}~${(i + 1) * 2048 - 1})`)
  }

  // let datas = []        // 表格中每一行的数据
  // featuresName.unshift('传感器')  // 表格的列名
  // for (const sensor in featuresGroupBySensor) {
  //   let featuresOfSensor = featuresGroupBySensor[sensor].slice()
  //   featuresOfSensor.unshift(sensor)
  //   datas.push(featuresOfSensor)
  // }
  // console.log("........datas: ", datas)

  // datas是每个传感器的每一帧样本所提取到的特征

  // 特征表格
  // columns.value.length = 0
  // 将特征名作为列名
  // featuresName.forEach(element => {
  //   columns.value.push({ prop: element, label: element, width: 180 })
  // });

  // 转换各特征值数据为对象数组，以作为表格数据进行显示
  // datas.forEach(data => {
  //   transformedData.value = data.map((row, index) => {
  //     const obj = {};
  //     columns.value.forEach((column, colIndex) => {
  //       obj[column.prop] = row[colIndex];
  //     });
  //     return obj;
  //   });
  // });  

  let rawDataSeries: any = resultsObject.raw_data
  numOfSensors.value = rawDataSeries.length
  console.log('rawDataSeries: ', rawDataSeries)
  // 原始信号波形图显示
  let sensorNo = 1
  rawDataList.value.length = 0
  for (let series of rawDataSeries) {
    rawDataList.value.push({
      'sensor_no': '传感器 ' + sensorNo,
      'data': series,
    })
    sensorNo += 1
  }
  sensorNo = 1
  featuresSeriesList.value.length = 0
  for (let features of Object.values(featuresToDrawLineChart)) {
    featuresSeriesList.value.push({
      'sensor_no': '传感器 ' + sensorNo,
      'data': features,
    })
    sensorNo += 1
  }
  console.log('featuresSeriesList: ', featuresSeriesList.value)
  featuresExtractionRawData.value = rawDataList.value[0].sensor_no  //默认显示第一个传感器的原始信号

  displayFeatureExtraction.value = true  // 显示特征提取结果

  // console.log('length: ', num_sensors.value)
  nextTick(() => {
    // 使用echarts绘制特征提取的原始信号波形图
    rawDataList.value.forEach(object => {
      let chart = echarts.init(document.getElementById(object.sensor_no))
      let dataSeries = object.data
      let option = {
        title: {
          text: '原始信号'
        },
        xAxis: {
          type: 'value',
          data: Array.from({length: dataSeries.length}),
          name: '采样点'
        },
        yAxis: {
          type: 'value',
          name: '采样值'
        },
        series: [
          {
            name: '信号',
            type: 'line',
            symbol: 'circle',
            symbolSize: 2,
            data: dataSeries.map((value, index) => [index, value])
          }
        ]
      }
      chart.setOption(option)
    })
    featuresSeriesList.value.forEach(object => {
      // 使用echarts绘制特征提取的特征折线图
      type EChartsOption = echarts.EChartsOption;
      let dataSeries = object.data
      var lineChartDom = document.getElementById(object.sensor_no + 'features')
      var lineChart = echarts.init(lineChartDom);
      var lineChartOption: EChartsOption;

      lineChartOption = {
        title: {
          text: '连续信号样本提取特征'
        },
        tooltip: {
          trigger: 'axis'
        },
        legend: {
          // data: ['Email', 'Union Ads', 'Video Ads', 'Direct', 'Search Engine']
          left: 'center',
          top: '5%',
          bottom: '6%',
          data: featuresName
        },
        grid: {
          left: '5%',
          right: '5%',
          bottom: '3%',
          containLabel: true
        },
        toolbox: {
          feature: {
            saveAsImage: {}
          }
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          // data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
          data: x_axis
        },
        yAxis: {
          type: 'value'
        },
        series: []
      };
      for (let key of featuresName) {
        lineChartOption.series.push({
          name: key,
          type: 'line',
          stack: 'Total',
          data: dataSeries[key]
        })
      }
      lineChart.setOption(lineChartOption);
    })
  })
}

// 特征选择结果可视化
const displayFeatureSelection = ref(false)
const featureSelectionFigure = ref('')
const featuresSelected = ref('')
const selectFeatureRule = ref('')
const correlationFigure = ref('')
const featuresSelectionTabs = ref('first')

const featuresSelectionDisplay = (resultsObject) => {
  displayFeatureSelection.value = true

  let figure1 = resultsObject.figure_Base64
  let figure2 = resultsObject.heatmap_Base64
  featuresSelected.value = resultsObject.selected_features.join('、')

  selectFeatureRule.value = resultsObject.rule
  featureSelectionFigure.value = 'data:image/png;base64,' + figure1
  correlationFigure.value = 'data:image/png;base64,' + figure2

}


// 用户对于故障诊断结果准确性
const feedBackDialogVisible = ref(false)
// const feedbackContent = ref('')
const feedBackFormRef = ref()

const feedBackRules = {
  selectedModule: [
    { required: true, message: '请选择一个模块', trigger: 'change' }
  ],
  feedbackContent: [
    { required: true, message: '请输入问题描述', trigger: 'blur' },
    { pattern: /^[\u4e00-\u9fa5a-zA-Z0-9_]+$/, message: '只能包含中英文字符、数字和下划线', trigger: 'blur' }
  ]
}


const feedBackFormRefState = reactive({
  module: '',
  feedbackContent: ''
})

const feedBack = () => {
  // console.log("feedBackFormRef: ", feedBackFormRef.value)
  // feedBackFormRef.value.validate.then(() => {
    
    const formData = new FormData();
    formData.append('username', username.value)
    formData.append('datafile', usingDatafile.value)
    formData.append('modelName', modelLoaded.value)
    formData.append('module', feedBackFormRefState.module)
    formData.append('question', feedBackFormRefState.feedbackContent)
    formData.append('modelId', modelLoadedId)
    api.post('user/user_feedback/', formData).then((response: any)=>{
        if(response.data.code === 200){

          message.success('已收到您的反馈')
        }else{
          message.error("反馈提交失败, ", response.data.message)
        }
      }
    )
    .catch(error => {
      console.log("上传反馈失败", error)
      message.error("上传反馈失败")
    })
//   })
//   .catch((error: any) => {
//     console.error('反馈提交失败', error);
//     message.error('反馈提交失败，请重试');
//   })
}

// 故障诊断结果展示
const displayFaultDiagnosis = ref(false)
const faultDiagnosis = ref('')
const faultDiagnosisFigure = ref('')
const faultDiagnosisResultsText = ref('')
const faultDiagnosisResultOption = ref('1')

const faultDiagnosisDisplay = (resultsObject: any) => {
  displayFaultDiagnosis.value = true

  let figure1 = resultsObject.figure_Base64
  let diagnosisResult = resultsObject.diagnosis_result
  let indicator = resultsObject.indicator
  let x_axis = resultsObject.x_axis
  let num_has_fault = resultsObject.num_has_fault
  let num_has_no_fault = resultsObject.num_has_no_fault

  console.log('indicator: ', indicator)
  // 获取indicator的key
  let indicatorKeys = Object.keys(indicator)

  nextTick(() => {

    // 绘制连续样本指标变化折线图
    type EChartsOption = echarts.EChartsOption;
    var lineChartDom = document.getElementById('indicatorVaryingFigure')!;
    var lineChart = echarts.init(lineChartDom);
    var lineChartOption: EChartsOption;

    lineChartOption = {
      title: {
        text: '连续样本指标变化曲线图'
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        // data: ['Email', 'Union Ads', 'Video Ads', 'Direct', 'Search Engine']
        left: 'center',
        top: '5%',
        bottom: '6%',
        data: indicatorKeys
      },
      grid: {
        left: '5%',
        right: '5%',
        bottom: '3%',
        containLabel: true
      },
      toolbox: {
        feature: {
          saveAsImage: {}
        }
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        // data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        data: x_axis
      },
      yAxis: {
        type: 'value'
      },
      series: []
    };

    for (let key in indicator) {
      lineChartOption.series.push({
        name: key,
        type: 'line',
        stack: 'Total',
        data: indicator[key]
      })
    }
    lineChart.setOption(lineChartOption);

    // 绘制故障样本与非故障样本数量饼状图
    var pieChartDom = document.getElementById('faultExampleRatioFigure');
    var pieChart = echarts.init(pieChartDom);
    var pieChartOption;

    pieChartOption = {
      title: {
        text: '预测结果中不同类型样本数量比例',
        // subtext: 'Fake Data',
        left: 'center'
      },
      tooltip: {
        trigger: 'item'
      },
      legend: {
        orient: 'vertical',
        left: 'left'
      },
      series: [
        {
          name: '样本数量',
          type: 'pie',
          radius: '50%',
          data: [
            {value: num_has_fault, name: '预测为故障类型的样本'},
            {value: num_has_no_fault, name: '预测为非故障类型的样本'},
            // { value: 580, name: 'Email' },
            // { value: 484, name: 'Union Ads' },
            // { value: 300, name: 'Video Ads' }
          ],
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
    };

    pieChart.setOption(pieChartOption);
  })


  faultDiagnosisResultsText.value = resultsObject.resultText
  if (diagnosisResult == 0) {
    faultDiagnosis.value = '无故障'
  } else {
    faultDiagnosis.value = '存在故障'
  }
  faultDiagnosisFigure.value = 'data:image/png;base64,' + figure1

}


// 故障预测结果展示
const displayFaultRegression = ref(false)
const haveFault = ref(0)
const faultRegression = ref('')
const timeToFault = ref('')
const faultRegressionFigure = ref('')

const faultRegressionDisplay = (resultsObject) => {
  displayFaultRegression.value = true

  let figure1 = resultsObject.figure_Base64
  faultRegressionFigure.value = 'data:image/png;base64,' + figure1
  // let fault_time = results_object.time_to_fault

  if (resultsObject.time_to_fault == 0) {
    haveFault.value = 1
    faultRegression.value = '已经出现故障'
  } else {
    haveFault.value = 0
    faultRegression.value = '还未出现故障'
    timeToFault.value = resultsObject.time_to_fault_str
  }

}

// 插值处理可视化
const activeName3 = ref('1')
const displayInterpolation = ref(false)
const interpolationFigures = ref([]) // 插值处理结果图像
const interpolationResultsOfSensors = ref([])   // 插值处理结果中有几个传感器

const interpolationDisplay = (resultsObject: any) => {
  displayInterpolation.value = true

  let sensorId = 0
  interpolationFigures.value.length = 0
  interpolationResultsOfSensors.value.length = 0
  for (const [key, value] of Object.entries(resultsObject)) {
    sensorId += 1
    interpolationFigures.value.push('data:image/png;base64,' + value)
    interpolationResultsOfSensors.value.push({label: key.split('_')[0], name: sensorId.toString()})
  }
  // console.log('interpolationResultsOfSensors: ', interpolationResultsOfSensors)
  // console.log('interpolationFigures: ', interpolationFigures)
  // displayDenoise.value = true
}

// 无量纲化可视化
const activeName4 = ref('1')
const displayNormalization = ref(false)
const normalizationFormdataRaw = ref([])
const normalizationFormdataScaled = ref([])  // 无量纲的结果表格
const normalizationColumns = ref([])
const normalizationResultFigures = ref([])   // 无量纲结果图像
const normalizationResultsSensors = ref([])

const transformDataToFormdata = (features_with_name: any, columns: any, formdata: any) => {
  // 从后端返回的结果中提取出特征名称和特征值，并转化为表格数据
  let featuresName = features_with_name.features_name.slice()
  let featuresGroupBySensor = Object.assign(features_with_name.features_extracted_group_by_sensor)
  let datas = []        // 表格中每一行的数据
  featuresName.unshift('传感器')  // 表格的列名
  for (const sensor in featuresGroupBySensor) {
    let features_of_sensor = featuresGroupBySensor[sensor].slice()
    features_of_sensor.unshift(sensor)
    datas.push(features_of_sensor)
  }

  columns.value.length = 0
  featuresName.forEach(element => {
    columns.value.push({prop: element, label: element, width: 180})
  });

  // 转换数据为对象数组  
  formdata.value = datas.map((row, index) => {
    const obj = {};
    columns.value.forEach((column, colIndex) => {
      obj[column.prop] = row[colIndex];
    });
    return obj;
  });
}

const normalizationResultType = ref('table')   // 无量纲化的结果类型，table表示表格，figure表示图像

const normalizationDisplay = (resultsObject: any) => {
  displayNormalization.value = true

  let rawData = Object.assign({}, resultsObject.raw_data)
  let scaledData = Object.assign({}, resultsObject.scaled_data)

  // 无量纲化结果为表格数据
  if (resultsObject.datatype == 'table') {
    normalizationResultType.value = 'table'
    transformDataToFormdata(rawData, normalizationColumns, normalizationFormdataRaw)
    transformDataToFormdata(scaledData, normalizationColumns, normalizationFormdataScaled)
  }
  // 无量纲化的对象为信号序列，结果为信号序列的波形图
  else {
    normalizationResultType.value = 'figure'
    let sensorId = 0
    normalizationResultFigures.value.length = 0
    normalizationResultsSensors.value.length = 0
    for (const [key, value] of Object.entries(resultsObject)) {
      if (key == 'datatype') {
        continue
      }
      sensorId += 1
      normalizationResultFigures.value.push('data:image/png;base64,' + value)
      normalizationResultsSensors.value.push({label: key.split('_')[0], name: sensorId.toString()})
    }
  }
}

// 小波降噪可视化
const activeName2 = ref('1')
const displayDenoise = ref(false)
const denoiseFigures = ref([])  // 存放小波降噪结果图片
const waveletResultsOfSensors = ref([])  // 存放不同传感器的小波降噪结果

const denoiseDisplay = (resultsObject) => {
  console.log('results_object: ', resultsObject)
  let sensorId = 0
  denoiseFigures.value.length = 0
  waveletResultsOfSensors.value.length = 0
  for (const [key, value] of Object.entries(resultsObject)) {
    sensorId += 1
    denoiseFigures.value.push('data:image/png;base64,' + value)
    waveletResultsOfSensors.value.push({label: key.split('_')[0], name: sensorId.toString()})
  }
  console.log('results_of_sensors: ', waveletResultsOfSensors)
  console.log('denoiseFigures: ', denoiseFigures)
  displayDenoise.value = true
}

// 清除可视化区域
const resultsViewClear = () => {
  showPlainIntroduction.value = false  // 清除算法介绍
  showStatusMessage.value = false      // 清除程序运行状态
  canShowResults.value = false         // 清除可视化区域元素
  contrastVisible.value = false    // 清除
  show1.value = true
  loading.value = true
  isShow.value = false
  // 清除所有结果可视化
  displayHealthEvaluation.value = false
  displayFeatureExtraction.value = false
  displayFeatureSelection.value = false
  displayFaultDiagnosis.value = false
  displayFaultRegression.value = false
  displayInterpolation.value = false
  displayNormalization.value = false
  displayDenoise.value = false
  displayRawDataWaveform.value = false

  if (!done.value) {
    currentDisplayedItem = ''
  }
}


const displayRawDataWaveform = ref(false)
const rawDataWaveform = ref('')
const currentDataBrowsing = ref('')
// 用户浏览原始数据
const browseDataset = (row: { dataset_name: any; }) => {

  // 清除可视化区域内容
  resultsViewClear()
  canShowResults.value = true
  // 发送请求获取原始数据的波形图
  let filename = row.dataset_name

  api.get('user/browse_datafile/?filename=' + filename).then((response: any) => {
    if (response.status === 200) {
      displayRawDataWaveform.value = true
      let data = response.data
      let figure = data.figure_Base64

      rawDataWaveform.value = 'data:image/png;base64,' + figure
      currentDataBrowsing.value = filename

      console.log('访问成功：')
    } else {
      ElMessage.error('访问文件失败')
    }
  })
      .catch((error: any) => {
        console.log('访问文件失败：', error)
      })
}

// 当前显示的算法模块结果
let currentDisplayedItem = ''

// 点击可视化建模区中的算法模块显示对应的结果
const showResult = (item) => {

  if (done.value) {

    if (missionComplete.value) {
      if (item.label == '数据源') {
        ElMessage.info('数据源模块没有结果可显示')
        return
      }
      if (item.label != '层次逻辑回归评估' && item.label != '层次分析模糊综合评估' && item.label != '层次朴素贝叶斯评估' && item.label != '特征提取' && item.label != '特征选择' && item.label != '故障诊断'
          && item.label != '故障预测' && item.label != '特征提取' && item.label != '插值处理' && item.label != '无量纲化' && item.label != '小波变换' && item.label != '健康评估'
      ) {
        showPlainIntroduction.value = false
        showStatusMessage.value = false
        show1.value = true
        loading.value = true
        canShowResults.value = false
        isShow.value = false
        setTimeout(function () {
          isShow.value = true
          show1.value = false
          loading.value = false
        }, 2500);
        let moduleName = item.label
        let url = 'http://127.0.0.1:8000/homepage?display=' + moduleName
        axios.request({
          method: 'GET',
          url: url,
        });
        setTimeout(function () {
          // 为 iframe 的 src 属性添加一个查询参数，比如当前的时间戳，以强制刷新
          var iframe = document.getElementById('my_gradio_app');
          var currentSrc = iframe.src;
          var newSrc = currentSrc.split('?')[0]; // 移除旧的查询参数
          iframe.src = newSrc + '?updated=' + new Date().getTime();
        }, 2400);
      } else {
        resultsViewClear()
        canShowResults.value = true
        if (item.label == '层次分析模糊综合评估') {
          let results_to_show = responseResults.层次分析模糊综合评估
          if (currentDisplayedItem != '层次分析模糊综合评估') {
            currentDisplayedItem = '层次分析模糊综合评估'
          } else {
            displayHealthEvaluation.value = true  // 显示健康评估结果
            return
          }
          healthEvaluationDisplay(results_to_show)
        } else if (item.label == '特征提取') {
          if (currentDisplayedItem != '特征提取') {
            currentDisplayedItem = '特征提取'
          } else {
            displayFeatureExtraction.value = true  // 显示特征提取结果
            return
          }
          let results_to_show = responseResults.特征提取
          featureExtractionDisplay(results_to_show)
        } else if (item.label == '特征选择') {
          if (currentDisplayedItem != '特征选择') {
            currentDisplayedItem = '特征选择'
          } else {
            displayFeatureSelection.value = true  // 显示特征选择结果
            return
          }
          let results_to_show = responseResults.特征选择
          featuresSelectionDisplay(results_to_show)
        } else if (item.label == '故障诊断') {
          if (currentDisplayedItem != '故障诊断') {
            currentDisplayedItem = '故障诊断'
          } else {
            displayFaultDiagnosis.value = true
            return
          }
          let results_to_show = responseResults.故障诊断
          faultDiagnosisDisplay(results_to_show)
        } else if (item.label == '故障预测') {
          let results_to_show = responseResults.故障预测
          faultRegressionDisplay(results_to_show)
        } else if (item.label == '插值处理') {
          let results_to_show = responseResults.插值处理
          interpolationDisplay(results_to_show)
        } else if (item.label == '无量纲化') {
          let results_to_show = responseResults.无量纲化
          normalizationDisplay(results_to_show)
        } else if (item.label == '小波变换') {
          let results_to_show = responseResults.小波变换
          denoiseDisplay(results_to_show)
        } else if (item.label == '层次朴素贝叶斯评估') {
          let results_to_show = responseResults.层次朴素贝叶斯评估
          if (currentDisplayedItem != '层次朴素贝叶斯评估') {
            currentDisplayedItem = '层次朴素贝叶斯评估'
          } else {
            displayHealthEvaluation.value = true
            return
          }
          healthEvaluationDisplay(results_to_show)
        } else if (item.label == '层次逻辑回归评估') {
          let results_to_show = responseResults.层次逻辑回归评估
          if (currentDisplayedItem != '层次逻辑回归评估') {
            currentDisplayedItem = '层次逻辑回归评估'
          } else {
            displayHealthEvaluation.value = true
            return
          }
          healthEvaluationDisplay(results_to_show)
        } else if (item.label == '健康评估') {
          let results_to_show = responseResults.健康评估
          if (currentDisplayedItem != '健康评估') {
            currentDisplayedItem = '健康评估'
          } else {
            displayHealthEvaluation.value = true
            return
          }
          healthEvaluationDisplay(results_to_show)
        } else {
          ElMessage({
            message: '无效的算法模块',
            type: 'error'
          })
        }
      }
    }
  } else {
    // ElMessage({
    //   message: '当前无运行结果',
    //   type: 'error'
    // })
  }
}


// 从后端获取到的历史模型的信息
const fetchedModelsInfo = ref([])

// 打开抽屉，同时从后端获取历史模型
const fetchModels = () => {
  dataDrawer.value = false  // 打开历史模型抽屉

  // 向后端发送请求获取用户的历史模型
  api.get('/user/fetch_models/').then((response: any) => {
    if (response.data.code == 200) {
      modelsDrawer.value = true
      let modelsInfo = response.data.message
      fetchedModelsInfo.value.length = 0
      for (let item of modelsInfo) {
        fetchedModelsInfo.value.push(item)
      }
    }
    if (response.data.code == 401) {
      ElMessageBox.alert('登录状态已失效，请重新登陆', '提示',
          {
            confirmButtonText: '确定',
            callback: (action: Action) => {
              router.push('/')
            }
          }
      )
    }

  })
      .catch(error => {
        ElMessage({
          message: '获取历史模型失败,' + error,
          type: 'error'
        })
      })
}


//全屏
const toggleFullscreen = () => {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    document.documentElement.requestFullscreen();
  }
}


// 复用历史模型，不需要进行模型检查等操作
let modelHasBeenSaved = false
const modelLoaded = ref('无')  // 已加载的历史模型
let modelLoadedId: string  // 所加载的模型的唯一标识符

// 点击历史模型表格中使用按钮复现用户历史模型
const useModel = (row) => {

  // console.log('点击历史模型复现：', row)

  if (nodeList.value.length != 0) {
    nodeList.value.length = 0
  }
  handleClear()
  updateStatus('当前模型已保存')
  modelHasBeenSaved = true
  canStartProcess.value = false
  modelLoaded.value = row.model_name
  modelLoadedId = String(row.id)
  let objects
  try {
    objects = JSON.parse(row.model_info)
  } catch {
    objects = row.model_info
  }


  // if (typeof objects == 'string')

  let connection = objects.connection      // 模型连线信息
  let nodeList1 = objects.nodeList         // 模型节点信息

  console.log('connection:', connection)
  console.log('nodeList:', nodeList1)


  // 恢复节点
  for (let node of nodeList1) {

    nodeList.value.push(node)

    if (node.label == '特征提取') {
      features.value.length = 0
      let params = node.parameters[node.use_algorithm]
      for (let [key, value] of Object.entries(params)) {
        if (value) {
          features.value.push(key)
        }
      }
    }
  }
  // 用于将节点的id与节点的label对应起来
  let idToLabelList = {'nodeId': [], 'nodeLabel': []}

  // 初始化每个节点的可连接状态
  for (let node of nodeList.value) {

    let nodeId = node.id
    idToLabelList.nodeId.push(nodeId)
    idToLabelList.nodeLabel.push(node.label)

    nextTick(() => {
      // plumbIns.draggable(nodeId, { containment: "efContainer" })
      if (node.id === '2.2') {
        plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)
        return
      }
      plumbIns.makeSource(nodeId, deff.jsplumbSourceOptions)
      // plumbIns.addEndpoint(nodeId, deff.jsplumbTargetOptions)
      if (node.id === '1' || node.id === '4') {
        return
      }
      plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)
    })
  }

  // 根据返回的模型的连接顺序，恢复模型中的连线
  let connectionList = []
  let connection2 = []   // 记录每个节点的id
  let node_num = connection.length

  // 初始化connection2，用于记录每个节点的id
  for (let i = 0; i < node_num; i++) {
    let label = connection[i]
    for (let j = 0; j < node_num; j++) {
      if (idToLabelList.nodeLabel[j] === label) {
        connection2[i] = idToLabelList.nodeId[j]
        break
      }
    }
  }
  console.log("connection2:", connection2)

  saveModelSetting(false, connection)
  contentJson.schedule = connection
  modelSetup.value = true
  // 如果只有一个节点，则不恢复连线，否则按照模型信息中各模块的连接顺序恢复连线
  if (node_num == 1) {
    connectionList = []
  } else {
    for (let i = 0; i < node_num - 1; i++) {
      connectionList.push({'soruce_id': connection2[i], 'target_id': connection2[i + 1]})
    }
    nextTick(() => {
      for (let line of connectionList) {
        plumbIns.connect({
          source: document.getElementById(line.soruce_id),
          target: document.getElementById(line.target_id)
        })
      }
    })
  }
}


let index = 0
let row = 0
const deleteModelConfirmVisible = ref(false)
// 删除模型操作
const deleteModel = (indexIn, rowIn) => {
  index = indexIn
  row = rowIn
  deleteModelConfirmVisible.value = true
}
// 用户删除模型操作确认
const deleteModelConfirm = () => {

  // 发送删除请求到后端，row 是要删除的数据行
  api.get('/user/delete_model/?row_id=' + row.id).then((response) => {
    if (response.data.code == 401) {
      ElMessageBox.alert('登录状态失效，请重新登陆', '提示',
          {
            confirmButtonText: '确定',
            callback: (action: Action) => {
              router.push('/')
            }
          }
      )
    }
    if (response.data.code == 200) {
      // 如果被删除的模型已经被加载，则需要取消加载
      if (modelLoaded.value == row.model_name) {
        modelLoaded.value = '无'
        modelHasBeenSaved = false
        canStartProcess.value = true
        handleClear()
      }
      if (index !== -1) {
        // 删除前端表中数据
        fetchedModelsInfo.value.splice(index, 1)
        deleteModelConfirmVisible.value = false
        ElMessage({
          message: '删除模型成功',
          type: 'success'
        })
      }

    } else {
      if (response.data.code == 404) {
        ElMessage({
          message: '没有权限删除该模型',
          type: 'error'
        })
      } else {
        ElMessage({
          message: '删除模型失败，请稍后重试',
          type: 'error'
        })
      }
    }
  }).catch(error => {
    // 处理错误
    console.error(error);
    ElMessage({
      message: '删除模型失败,' + error,
      type: 'error'
    })
  });
}

// 查看模型的具体信息，按如下方式构造信息卡片
const modelName = ref('')
const modelAlgorithms = ref([])
const modelParams = ref([])  // {'模块名': xx, '算法': xx, '参数': xx}

const showModelInfo = (row) => {
  let objects = JSON.parse(row.model_info)
  let nodesList = objects.nodeList         // 模型节点信息   
  let connection = objects.connection     // 模型连接顺序

  modelName.value = row.model_name
  modelAlgorithms.value = connection
  modelParams.value.length = 0
  nodesList.forEach(element => {
    let item = {'模块名': '', '算法': ''}
    item.模块名 = element.label
    item.算法 = labelsForAlgorithms[element.use_algorithm]
    modelParams.value.push(item)
  });
}

// 用于显示程序运行的状态信息
const showStatusMessage = ref(false)
const statusMessageToShow = ref('')

// 程序运行状态信息
const statusMessage = {
  'success': '## 程序已经运行完毕，请点击相应的算法模块查看对应结果！',
  'shutdown': '## 程序运行终止，点击清空模型重新建立模型',
  'error': '## 程序运行出错，请检查模型是否正确，或者检查加载的数据是否规范，点击清空模型重新建立模型',
}


// 控制是否可以修改模型，值为true时，可以修改模型，值为false时，不能修改模型
const toRectifyModel = ref(false)

// 完成建模
const finishModeling = () => {
  if (nodeList.value.length) {
    if (linkedList.length() == 0 && nodeList.value.length == 1) {
      ElMessage({
        message: '完成建模',
        type: 'success'
      })
      canCheckModel.value = false
      modelSetup.value = true     // 不能删除建模区的模块
      done.value = true           // 不能拖动模块
      toRectifyModel.value = true // 可以点击修改模型进行修改模型
      updateStatus('模型建立完成，请点击模型检查')
      return
    }
    if (linkedList.length() != nodeList.value.length) {
      ElMessage({
        message: '请确保图中所有模块均已建立连接，且没有多余的模块',
        type: 'warning'
      })
      return
    }
  }

  ElMessage({
    message: '完成建模',
    type: 'success'
  })
  modelSetup.value = true     // 不能删除建模区模块
  done.value = true     // 不能拖动模块
  toRectifyModel.value = true // 可以修改模型
  canCheckModel.value = false
  updateStatus('模型建立完成，请点击模型检查')
}

// 修改模型
const rectifyModel = () => {
  canCheckModel.value = true
  canSaveModel.value = true
  canStartProcess.value = true
  canShutdown.value = true
  modelSetup.value = false     // 可以删除建模区模块
  done.value = false     // 可以拖动模块
  toRectifyModel.value = false
  ElMessage({
    showClose: true,
    message: '进行模型修改, 完成修改后请再次点击完成建模',
    type: 'info'
  })
  updateStatus('正在修改模型')
}

//检查模型
const checkModeling = () => {
  if (nodeList.value.length == 0 && !modelHasBeenSaved) {
    canCheckModel.value = true
  }
}

//保存模型
const saveModeling = () => {
  if (nodeList.value.length == 0 || modelHasBeenSaved) {
    canSaveModel.value = true
  }
}

//开始建模
const startModeling = () => {
  if (nodeList.value.length == 0) {
    canStartProcess.value = true
  }
}


// 建模状态更新
function updateStatus(status) {
  var indicator = document.getElementById('statusIndicator');
  indicator.textContent = status; // 更新文本  
  indicator.classList.remove('error', 'success', 'saved', 'rectify'); // 移除之前的状态类  
  switch (status) {
    case '未建立模型':
      // 默认样式，或者设置为特定类  
      break;
    case '模型建立完成，请点击模型检查':
      indicator.classList.add('error');
      break;
    case '模型建立并已通过模型检查':
      indicator.classList.add('success');
      break;
    case '当前模型已保存':
      indicator.classList.add('saved');
    case '正在修改模型':
      indicator.classList.add('rectify')
      break;
  }
}


const fetchedDataFiles = ref<Object[]>([])

// 用户目前选择的数据文件
const usingDatafile = ref('无')

const deleteDatasetConfirmVisible = ref(false)
let rowDataset: any = null
let indexDataset: any = null


// 用户删除历史数据
// const deleteDataset = (index_in: any, row_in: any) => {
//   indexDataset = index_in
//   rowDataset = row_in
//   deleteDatasetConfirmVisible.value = true
// }
// 确认删除历史数据文件
const deleteDatasetConfirm = (index: any, row: any) => {

  api.get('/user/delete_datafile/?filename=' + row.dataset_name)
      .then((response: any) => {
        if (response.data.code == 200) {
          // 删除前端表中数据
          fetchedDataFiles.value.splice(index, 1)
          deleteDatasetConfirmVisible.value = false
          ElMessage({
            message: '文件删除成功',
            type: 'success'
          })
          // 如果文件已经被加载，则需要取消加载行为
          if (rowDataset.dataset_name == usingDatafile.value) {
            usingDatafile.value = '无'
          }
        } else if (response.data.code == 400) {
          ElMessage({
            message: '删除失败: ' + response.data.message,
            type: 'error'
          })
        } else if (response.data.code == 401) {
          ElMessageBox.alert('登录状态已失效，请重新登陆', '提示', {
            confirmButtonText: '确定',
            callback: (action: Action) => {
              router.push('/')
            },
          })
        }
      })
      .catch((error: any) => {
        // console.log('delete_datafile_error: ', error)
        // ElMessage({
        //   message: '删除失败',
        //   type: 'error'
        // })
      })
}


const loadingData = ref(false)


// 用户选择历史数据进行加载
const useDataset = (row_in: any) => {
  loadingData.value = true
  setTimeout(() => {
    loadingData.value = false
    usingDatafile.value = row_in.dataset_name
    ElMessage({
      message: '数据加载成功',
      type: 'success'
    })
  }, 1000)

}

// const handleSwitchDrawer = (fetchData: any[]) => {

//   modelsDrawer.value = false;

//   fetchedDataFiles.value = []
//   // fetchData.forEach(element => {
//   //   fetchedDataFiles.value.push(element)
//   // });
//   for (let item of fetchData){
//     fetchedDataFiles.value.push(item)
//   }

//   dataDrawer.value = true

//   // Object.assign(fetchedDataFiles, fetchData)
// };


// 已加载模型和已加载数据字体颜色更改
const getColor = (value: string) => {
  if (value == '无') {
    return 'red'
  } else {
    return 'green'
  }
}
</script>

<style>
body {
  margin: 0;
}


.item {
  width: 150px;
  height: 50px;
  position: relative;
}

.deleteButton {
  position: absolute;
  top: 2px;
  right: 2px;
}

#source {
  border: 2px solid red;
}

#target {
  border: 2px solid blue;
}

.main {
  display: flex;
}

ul {
  list-style: none;
  padding-left: 0;
  width: 120px;
  background: #eee;
  text-align: center;
}

ul > li {
  height: 40px;
  line-height: 40px;
}

.main-right {
  border: 1px solid #ccc;
  flex: 1;
  margin-left: 15px;
  position: relative;
  background: #f4f4f4;
}

.node-info {
  position: relative;
  top: 5px
}

.node-info-label {
  /* font-style: italic; */
  /* 垂直和水平居中的样式 */
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  white-space: normal;
  overflow: visible;
  /* font-style: italic; */
  padding: 4px;
  position: relative;
  width: 180px;
  height: 70px;
  /* line-height: 36px; */
  font-size: 12px;
  text-align: center;
  border: 1px solid #e5e7eb;
  background: #fff;

}

.node-info-label:hover {

  cursor: pointer;
  background: #f4eded;
}

.node-info-label:hover + .node-drag {
  /* background: red; */
  display: inline-block;
}

.node-drag {
  width: 10px;
  height: 10px;
  border-radius: 10px;
  background-color: gray;
  border: 1px solid #ccc;
  position: absolute;
  right: -7px;
  top: 50%;
  transform: translateY(-50%);
}

.node-drag:hover {
  display: inline-block;

}

.fullscreen_container {
  height: 100vh;
  /* display: flex; */
  /* flex-direction: column; */
}

.result_visualization_view {
  width: 1200px;
  height: 600px;
  position: absolute;
}

.demo-tabs .custom-tabs-label span {
  vertical-align: middle;
  font-size: 16px;
  margin-left: 9px;
}

.status-indicator {
  position: absolute;
  top: 10px;
  left: 10px;
  padding: 5px 10px;
  border-radius: 5px;
  border: solid 1px rgba(0, 0, 0, 0.2);
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  background-color: #f0ad4e;
  font-size: 22px;
  /* 初始颜色，如黄色 */
  color: white;
  z-index: 1000;
  /* 确保它显示在其他元素之上 */
}

/* 可以为不同的状态添加额外的类 */
.status-indicator.error {
  background-color: #dc4e11;
  /* 红色表示错误或未通过检查 */
}

.status-indicator.success {
  background-color: #5cb85c;
  /* 绿色表示成功 */
}

.status-indicator.saved {
  background-color: #337ab7;
  /* 蓝色表示已保存 */
}

.status-indicator.rectify {
  background-color: #48a4a3;
  /* 表示正在修改模型 */
}

html,
body {
  height: 100vh;
  margin: 0;
  /* 移除默认的边距 */
  padding: 0;
  /* 移除默认的内边距 */
}

.el-main {
  background-color: #CCD0D6;
  /*color: #333;*/
  text-align: center;
  position: relative;
  background-position: center;
  background-size: contain;
  /* height: 50vh; */
  background-repeat: no-repeat;
}

.has-background {
  background-color: #CCD0D6;
  /*color: #333;*/
  text-align: center;
  position: relative;
  background-image: url('../assets/modeling.png');
  background-position: center;
  background-size: contain;
  /* height: 50vh; */
  background-repeat: no-repeat;
}

.clickable:hover {
  cursor: pointer;
  color: #007BFF;
}

.aside-title {
  font-size: 20px;
  font-weight: 700;
  background-color: #ffffff;
  justify-content: center;
  display: flex;
  align-items: center;
  gap: 5px;
  border-top: solid 3px #7D8081;

  text-align: center;
  width: 250px;
  height: 50px;
  color: #093256;
}

.first-menu-item {
  width: 150px;
  margin-top: 10px;
  background-color: #4C74DA;
  color: white;
}

.second-menu-item {
  width: 150px;
  margin-top: 7px;
  background-color: #7E9CE6;
}

.third-menu-item {
  background-color: #B9BFCE;
  margin-top: 7px;
  width: 145px;
  height: 30px;
  margin-bottom: 10px;
  padding: 0px;
  border-radius: 5px;
  align-content: center;
  margin-left: 40px;
}

.el-icon-devops {
  background: url('../assets/userModels.svg') center no-repeat;
  font-size: 12px;
  background-size: cover;
}

/* .el-space {
  display: flex;
  justify-content: center;
  align-items: center;
} */

.operation-button {
  /* margin: 0 5px; */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.operation-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}


.menuList-First {
  width: 150px;
  margin-top: 10px;
  background-color: #063e66;
  color: white;
}

.menuList-Second {
  width: 150px;
  margin-top: 10px;
  background-color: #2869C7;
  color: white;
}

/* 修改单选按钮选中状态的颜色 */
.ant-radio-button-wrapper-checked {
  color: #fff;
  background-color: #da222a;
  border-color: #da222a;
}

/* 修改单选按钮未选中状态的颜色 */
.ant-radio-button-wrapper {
  border-color: #da222a; /* 可以改变未选中时的边框颜色 */
}

/* 修改单选按钮组内的按钮之间的分隔线颜色 */
.ant-radio-button-wrapper:not(:last-child)::after {
  background-color: #da222a;
}

/* 修改单选按钮悬停时的颜色 */
.ant-radio-button-wrapper:hover {
  color: #fff;
  background-color: #c71e26;
  border-color: #c71e26;
}

.text-button {
  background: none;
  border: none;
  color: #333;
  font-family: 'Arial', sans-serif;
  font-size: 18px;
  padding: 10px 20px;
  cursor: pointer;
  transition: color 0.3s, border-bottom 0.3s;
}

.text-button.active {
  color: #007bff;
  border-bottom: 2px solid #007bff;
}

.module-name {
  width: 100%; /* 设置固定的宽度 */
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.custom-radio-button:hover {
  background-color: #c4e8ba;

}
</style>

<style scoped lang="scss">
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: rgba(204, 208, 214);
  padding: 0 20px;
  height: 60px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  border-bottom: rgb(95, 117, 154) 1px solid;
  text-align: center;
  line-height: 60px;
  position: relative;
  margin-bottom: 3px;
  width: 100%;
}

.logo {
  display: flex;
  flex: 4;
  align-items: center;
  justify-content: flex-start;
  font-size: 22px;
}

.logo img {
  height: 36px;
  margin-right: 12px;
}

.logo-text {
  font-size: 24px;
  font-weight: bold;
  color: #409EFF;
}

.title {
  flex-grow: 1;
  display: flex;
  flex: 4;
  justify-content: center; /* 水平居中 */
  align-items: center; /* 垂直居中 */
}

.title h1 {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
  color: #333;
  font-family: 'Microsoft YaHei', sans-serif; /* 修改为微软雅黑字体 */
}

.user-actions {
  display: flex;
  flex: 4;
  align-items: center;
  justify-content: flex-end;
  margin-right: 0;
}

.welcome-message {
  font-size: 15px;
  color: #409EFF;
  margin-right: 18px;
}

.action-item {
  display: flex;
  align-items: center;
  margin-right: 18px;
  cursor: pointer;
}

.action-item .action-icon {
  font-size: 18px;
  margin-right: 5px;
  color: #409EFF;
}

.action-item .action-text {
  font-size: 14px;
  color: #409EFF;
}

.action-item:hover .action-icon,
.action-item:hover .action-text {
  color: #66b1ff;
}

.action-item:hover .action-icon {
  transform: scale(1.2);
  transition: transform 0.3s;
}
</style>
