// 算法名称
export const labelsForAlgorithms = {
    WhiteGaussianNoise: "高斯白噪声",
    polynomial_interpolation: "多项式插值",
    bicubic_interpolation: "三次样条插值",
    lagrange_interpolation: "拉格朗日插值",
    newton_interpolation: "牛顿插值",
    linear_interpolation: "线性插值",
    time_domain_features: "单传感器时域特征提取",
    frequency_domain_features: "单传感器频域特征提取",
    time_frequency_domain_features: "单传感器时域和频域特征提取",
    time_domain_features_multiple: "多传感器时域特征提取",
    frequency_domain_features_multiple: "多传感器频域特征提取",
    time_frequency_domain_features_multiple: "多传感器时域和频域特征提取",
    FAHP: "单传感器层次分析模糊综合评估法",
    FAHP_multiple: "多传感器层次分析模糊综合评估法",
    BHM: '单传感器层次朴素贝叶斯评估',
    BHM_multiple: '多传感器层次朴素贝叶斯评估',
    AHP: "单传感器层次逻辑回归评估法",
    AHP_multiple: "多传感器层次逻辑回归评估法",
    feature_imp: "单传感器树模型的特征选择",
    feature_imp_multiple: "多传感器树模型的特征选择",
    mutual_information_importance: "单传感器互信息重要性特征选择",
    mutual_information_importance_multiple: "多传感器互信息重要性特征选择",
    correlation_coefficient_importance: "单传感器相关系数重要性特征选择",
    correlation_coefficient_importance_multiple: "多传感器相关系数重要性特征选择",
    random_forest: "单传感器随机森林故障诊断",
    svc: "单传感器SVM的故障诊断",
    gru: "单传感器GRU的故障诊断",
    lstm: "单传感器LSTM的故障诊断",
    random_forest_multiple: "多传感器随机森林故障诊断",
    svc_multiple: "多传感器SVM的故障诊断",
    gru_multiple: "多传感器GRU的故障诊断",
    lstm_multiple: "多传感器LSTM的故障诊断",
    wavelet_trans_denoise: "小波变换降噪",
    max_min: "max_min标准化",
    "z-score": "z-score标准化",
    robust_scaler: "鲁棒标准化",
    max_abs_scaler: "最大绝对值标准化",
    neighboring_values_interpolation: "邻近值插值",
    linear_regression: "单传感器线性回归故障预测",
    linear_regression_multiple: "多传感器线性回归故障预测",
    deeplearning_interpolation: '深度学习插值',
    ulcnn: '单传感器一维卷积深度学习模型的故障诊断',
    ulcnn_multiple: '多传感器一维卷积深度学习模型的故障诊断',
    spectrumModel: '单传感器基于时频图的深度学习模型的故障诊断',
    spectrumModel_multiple: '多传感器基于时频图的深度学习模型的故障诊断',
    dataSource: '数据源',
    customModule: '自定义模块',
    private_interpolation: '增值服务组件（插值）',
    private_fault_diagnosis_deeplearning: '增值服务组件（深度学习故障诊断）',  // 私有深度学习故障诊断算法
    private_fault_diagnosis_machine_learning: '增值服务组件（机器学习故障诊断）',  // 私有机器学习故障诊断算法
    private_fault_prediction: '增值服务组件（故障预测）',  // 私有故障预测算法
    extra_feature_selection: '增值服务组件（特征选择）',  // 私有特征选择算法
    private_scaler: '增值服务组件（无量纲化）',   // 私有标准化算法
    extra_health_evaluation: '增值服务组件（健康评估）',  // 私有健康评估算法
    extra_wavelet_transform: '增值服务组件（小波变换）',
};

// 算法介绍
export const algorithmIntroduction = {
    WhiteGaussianNoise:
        "高斯白噪声(White Gaussian Noise)在通信、信号处理及科学研究等多个领域中扮演着重要角色。它作为一种理想的噪声模型，具有独特的统计特性和功率谱分布，为系统性能评估、算法测试及信号分析提供了有力工具",
    polynomial_interpolation:
        "多项式补插法是一种分段三次Hermite插值方法,它在每个数据段上使用三次多项式来逼近数据点,并且在连接处保持一阶导数的连续性。与双三次插值不同," +
        "多项式插值在每个段上使用不同的三次多项式,并且尝试保持二阶导数的变号,从而生成一个形状类似于原始数据的曲线。",
    bicubic_interpolation:
        "双三次插值是一种平滑插值方法,它通过三次多项式段来逼近数据点,并且在每个数据段的连接处保持一阶导数和二阶导数的连续性。" +
        "这种方法可以生成一个平滑的曲线,通过数据点,并且在数据点处具有连续的一阶和二阶导数。",
    lagrange_interpolation:
        "对于所给数据中每一行每一列空白的位置,取空白位置上下3个相邻的值作为输入依据," +
        "根据拉格朗日算法构建一个多项式函数,使得该多项式函数在取得的这些点上的值都为零," +
        "将空白位置的行值作为输入,计算出y值,替换原来的空白值,从而达到插值的效果。",
    newton_interpolation:
        "在牛顿插值法中,首先利用一组已知的数据点计算差商,再将差商带入插值公式f(x)。将所提供数据中的数据各属性值作为y,而将索引号定义为x," +
        "对于所给数据中每一行每一列空白的位置,取空白位置上下4个相邻的值作为输入依据," +
        "并计算差商再反向带入包含差商的插值公式,替换原来的空白值。",
    linear_interpolation:
        "在线性插值算法中,首先遍历数据中的每一行每一列，找到空值位置并获取相邻点的值," +
        "然后去除相邻的空值,并处理边界情况,计算插值结果,替换原来的空白值。",
    time_domain_features:
        "时域特征提取是一种从信号中直接提取其在时间轴上特性的技术。它主要用于从原始信号中提取出诸如幅度、频率、周期、波形等关键信息，以便于后续的信号处理和分析。",
    frequency_domain_features:
        "频域特征提取是将时域信号转换到频域后，提取其频域上的特征。它主要用于从复杂的时域信号中提取出与频率相关的有用信息。",
    time_frequency_domain_features:
        "提取信号的时域和频域特征，可以结合时域与频域特征各自的优点",
    FAHP:
        "层次分析模糊综合评估法是一种将模糊综合评价法和层次分析法相结合的评价方法，在体系评价、效能评估，系统优化等方面有着广泛的应用，是一种定性与定量相结合的评价模型，一般是先用层析分析法确定因素集，\
             然后用模糊综合评判确定评判效果。模糊法是在层次法之上，两者相互融合，对评价有着很好的可靠性",
};


export const contrastOfAlgorithm = {
    '插值处理':
        "### 插值处理模块中各个插值算法的比较 \n" +
        "| 插值算法   | 优点                                           | 缺点                                         | \n" +
        "|--------|----------------------------------------------|--------------------------------------------| \n" +
        "| 邻近值插值  | 计算量很小，算法简单，因此运算速度较快。                         | 插值效果较差，有明显的锯齿状，细节部分不清晰。                    | \n" +
        "| 三次样条插值 | 保留了分段低次插值多项式的各种优点，同时提高了插值函数的光滑性。             | 需要更多的已知节点信息来保证插值结果的准确性，计算量较大。              | \n" +
        "| 拉格朗日插值 | 具有全局性，可以使用所有已知数据点进行插值，对于整个数据集的变化趋势能够较好地进行拟合。 | 计算复杂度高，需要计算每个数据点对应的拉格朗日基函数，尤其在数据点较多时计算量较大。 | \n" +
        "| 牛顿插值   | 计算简单，运算速度快。                                  | 在数据点分布不均匀或存在较大噪声时，可能无法给出准确的预测。             | \n" +
        "| 线性插值   | 计算简单，稳定性好，收敛性有保证。                            | 只能保证各小段曲线在连接点的连续性，但无法保证整条曲线的光滑性。           | \n" +
        "| 深度学习插值 | 能够处理复杂的非线性关系，对于大规模数据集有很好的适应性。插值精度高。          | 需要大量的训练数据和计算资源，训练过程可能比较复杂和耗时。              | \n"
    ,
    '特征提取':
        "### 特征提取模块中不同类型特征的比较 \n" +
        "| 特征类型    | 优点                             | 缺点                        |\n" +
        "|---------|--------------------------------|---------------------------|\n" +
        "| 时域特征    | 由时序信号直接得到，相对直观。计算简单，可反映总体状态。   | 作为故障诊断的依据，缺乏早期报警能力，易受噪声干扰。          |\n" +
        "| 频域特征    | 能够精确地反映信号的频率成分，能察觉微小的故障。       | 计算相对复杂。作为故障诊断的依据时，对于非平稳信号的诊断结果可能不准确。 |\n" +
        "| 时域和频域特征 | 结合时域特征和频域特征各自的特点，提供更全面的故障诊断依据。 | 计算复杂。                     |\n",
    '无量纲化':
        "### 无量纲化模块中不同标准化方法的比较 \n" +
        "| 标准化方法      | 定义                                    | 特点                                             |\n" +
        "|------------|---------------------------------------|------------------------------------------------|\n" +
        "| max-min标准化 | 将数据按照最小值和最大值进行线性变换，使其范围映射到[0,1]之间     | 简单易行，能够直接将数据缩放到特定范围，但可能导致数据分布不均匀时的失真           |\n" +
        "| z-score标准化 | 将数据减去平均值，再除以标准差，使得数据符合标准正态分布          | 保留了数据的原始分布，但改变了数据的尺度。适用于数据分布较为均匀的情况，对异常值较为敏感   |\n" +
        "| 鲁棒标准化      | 一种针对离群点做标准化处理的方法，对数据中心化和数据的缩放具有更强的鲁棒性 | 对异常值有较好的鲁棒性，能够减少异常值对标准化结果的影响，适用于数据集中存在较多异常值的情况 |\n" +
        "| 最大绝对值标准化   | 将数据除以数据的最大绝对值，使数据值落在[-1,1]之间          | 不会破坏原有数据的分布结构，不会破坏原有数据的分布结构                    |\n",
    '故障诊断':
        "### 故障诊断模块中不同算法的比较 \n" +
        "| 故障诊断算法   | 优点                         | 缺点              |\n"+
        "|------------|----------------------------|-----------------|\n"+
        "| 随机森林的故障诊断  | 处理高维数据能力强，抗过拟合能力强，对数据缺失不敏感 | 模型解释性较差，计算资源消耗大 |\n"+
        "| 支持向量机的故障诊断 | 处理非线性问题能力强，适于解决分类问题        | 计算复杂度高，对噪声敏感    |\n"+ 
        // "| 深度学习故障诊断        | 优点                    | 缺点          |\n"+
        // "|-----------------|-----------------------|-------------|\n"+
        "| GRU的故障诊断        | 时序建模能力强，泛化能力好         | 计算量大，对超参数敏感 |\n"+
        "| LSTM的故障诊断       | 能有效捕捉长序列语义关联          | 计算复杂度高      |\n"+
        "| 一维卷积的故障诊断       | 对序列数据中的局部模式敏感，计算复杂度较低 | 泛化能力较差      |\n"+
        "| 对于时频图的二维卷积的故障诊断 | 相较于一维卷积的故障诊断准确率较高     | 计算复杂度较高     |\n",
    '特征选择':
        "| 特征选择方法     | 优点                                              | 缺点                                                 |\n" +
        "|------------|-------------------------------------------------|----------------------------------------------------|\n" +
        "| 基于树模型的特征选择 | 预测逻辑易于理解，能够处理连续数据、离散数据以及缺失值等                    | 使用贪婪启发式方法进行训练，可能导致树不是全局最优的                         |\n" +
        "| 互信息法       | 互信息可以同时考虑特征与目标变量之间的相关性和特征之间的相关性，从而选择出具有较高区分度的特征 | 需要计算特征与目标变量之间的联合概率分布，计算复杂度较高。其次互信息法对数据集中的噪声和冗余特征敏感 |\n" +
        "| 相关系数法      | 相关系数是衡量两个变量之间线性关系强弱的指标，计算相对简单                   | 相关系数只能检测特征的线性关系，无法检测非线性关系                          |\n",

}


// 预定义的示例模型
export const predefinedModel = {
    "templateModel1": {
        "model_name": "用例模型1", "model_info": {
            "nodeList": [{ "label_display": "数据源", "label": "数据源", "id": "4", "nodeId": "4", "nodeContainerStyle": { "left": "77px", "top": "50px" }, "use_algorithm": "dataSource", "parameters": { "dataSource": {} }, "optional": true }, {
                "label_display": "单传感器时域和频域特征提取", "label": "特征提取", "id": "1.2", "nodeId": "1.2",
                "nodeContainerStyle": { "left": "297px", "top": "50px" }, "use_algorithm": "time_frequency_domain_features", "parameters": {
                    "time_domain_features": {
                        "均值": false, "方差": false, "标准差": false, "偏度": false, "峰度": false, "四阶累积量": false, "六阶累积量": false, "最大值": false, "最小值": false, "中位数": false, "峰峰值": false, "整流平均值": false,
                        "均方根": false, "方根幅值": false, "波形因子": false, "峰值因子": false, "脉冲因子": false, "裕度因子": false
                    }, "frequency_domain_features": { "重心频率": false, "均方频率": false, "均方根频率": false, "频率方差": false, "频率标准差": false, "谱峭度的均值": false, "谱峭度的峰度": false }, "time_frequency_domain_features": {
                        "均值": true, "方差": true, "标准差": true,
                        "峰度": true, "偏度": true, "四阶累积量": true, "六阶累积量": true, "最大值": true, "最小值": true, "中位数": true, "峰峰值": true, "整流平均值": true, "均方根": true, "方根幅值": true, "波形因子": true, "峰值因子": true, "脉冲因子": true, "裕度因子": true, "重心频率": true, "均方频率": true, "均方根频率": true, "频率方差": true, "频率标准差": true, "谱峭度的均值": true, "谱峭度的峰度": true
                    },
                    "time_domain_features_multiple": { "均值": false, "方差": false, "标准差": false, "偏度": false, "峰度": false, "四阶累积量": false, "六阶累积量": false, "最大值": false, "最小值": false, "中位数": false, "峰峰值": false, "整流平均值": false, "均方根": false, "方根幅值": false, "波形因子": false, "峰值因子": false, "脉冲因子": false, "裕度因子": false },
                    "frequency_domain_features_multiple": { "重心频率": false, "均方频率": false, "均方根频率": false, "频率方差": false, "频率标准差": false, "谱峭度的均值": false, "谱峭度的峰度": false }, "time_frequency_domain_features_multiple": {
                        "均值": false, "方差": false, "标准差": false, "峰度": false, "偏度": false, "四阶累积量": false, "六阶累积量": false, "最大值": false, "最小值": false,
                        "中位数": false, "峰峰值": false, "整流平均值": false, "均方根": false, "方根幅值": false, "波形因子": false, "峰值因子": false, "脉冲因子": false, "裕度因子": false, "重心频率": false, "均方频率": false, "均方根频率": false, "频率方差": false, "频率标准差": false, "谱峭度的均值": false, "谱峭度的峰度": false
                    }
                }, "optional": true
            },
            {
                "label_display": "单传感器信息重要性特征选择", "label": "特征选择", "id": "1.3", "nodeId": "1.3", "nodeContainerStyle": { "left": "502px", "top": "50px" }, "use_algorithm": "mutual_information_importance", "parameters": {
                    "feature_imp": { "rule": 1, "threshold1": null, "threshold2": null }, "mutual_information_importance": { "rule": 1, "threshold1": 0.5, "threshold2": null },
                    "correlation_coefficient_importance": { "rule": 1, "threshold": 0.005 }, "feature_imp_multiple": { "rule": 1, "threshold": 0.005 }, "mutual_information_importance_multiple": { "rule": 1, "threshold": 0.005 }, "correlation_coefficient_importance_multiple": { "rule": 1, "threshold": 0.005 }
                }, "optional": true
            },
            { "label_display": "单传感器随机森林故障诊断", "label": "故障诊断", "id": "2.1", "nodeId": "2.1", "nodeContainerStyle": { "left": "726px", "top": "50px" }, "use_algorithm": "random_forest", "parameters": { "random_forest": {}, "svc": {}, "gru": {}, "lstm": {}, "random_forest_multiple": {}, "svc_multiple": {}, "gru_multiple": {}, "lstm_multiple": {}, "ulcnn": {}, "ulcnn_multiple": {}, "spectrumModel": {}, "spectrumModel_multiple": {} }, "optional": false },
            { "label_display": "单传感器层次分析模糊综合评估法", "label": "层次分析模糊综合评估", "id": "3.1", "nodeId": "3.1", "nodeContainerStyle": { "left": "980px", "top": "50px" }, "use_algorithm": "FAHP", "parameters": { "FAHP": {}, "FAHP_multiple": {} }, "optional": false }], "connection": ["数据源", "特征提取", "特征选择", "故障诊断", "层次分析模糊综合评估"]
        }
    },
    "templateModel2": {
        "model_name": "用例模型2", "model_info": {
            "nodeList": [{ "label_display": "数据源", "label": "数据源", "id": "4", "nodeId": "4", "nodeContainerStyle": { "left": "77px", "top": "50px" }, "use_algorithm": "dataSource", "parameters": { "dataSource": {} }, "optional": true }, {
                "label_display": "单传感器时域和频域特征提取", "label": "特征提取", "id": "1.2", "nodeId": "1.2",
                "nodeContainerStyle": { "left": "297px", "top": "50px" }, "use_algorithm": "time_frequency_domain_features", "parameters": {
                    "time_domain_features": {
                        "均值": false, "方差": false, "标准差": false, "偏度": false, "峰度": false, "四阶累积量": false, "六阶累积量": false, "最大值": false, "最小值": false, "中位数": false, "峰峰值": false, "整流平均值": false,
                        "均方根": false, "方根幅值": false, "波形因子": false, "峰值因子": false, "脉冲因子": false, "裕度因子": false
                    }, "frequency_domain_features": { "重心频率": false, "均方频率": false, "均方根频率": false, "频率方差": false, "频率标准差": false, "谱峭度的均值": false, "谱峭度的峰度": false }, "time_frequency_domain_features": {
                        "均值": true, "方差": true, "标准差": true,
                        "峰度": true, "偏度": true, "四阶累积量": true, "六阶累积量": true, "最大值": true, "最小值": true, "中位数": true, "峰峰值": true, "整流平均值": true, "均方根": true, "方根幅值": true, "波形因子": true, "峰值因子": true, "脉冲因子": true, "裕度因子": true, "重心频率": true, "均方频率": true, "均方根频率": true, "频率方差": true, "频率标准差": true, "谱峭度的均值": true, "谱峭度的峰度": true
                    },
                    "time_domain_features_multiple": { "均值": false, "方差": false, "标准差": false, "偏度": false, "峰度": false, "四阶累积量": false, "六阶累积量": false, "最大值": false, "最小值": false, "中位数": false, "峰峰值": false, "整流平均值": false, "均方根": false, "方根幅值": false, "波形因子": false, "峰值因子": false, "脉冲因子": false, "裕度因子": false },
                    "frequency_domain_features_multiple": { "重心频率": false, "均方频率": false, "均方根频率": false, "频率方差": false, "频率标准差": false, "谱峭度的均值": false, "谱峭度的峰度": false }, "time_frequency_domain_features_multiple": {
                        "均值": false, "方差": false, "标准差": false, "峰度": false, "偏度": false, "四阶累积量": false, "六阶累积量": false, "最大值": false, "最小值": false,
                        "中位数": false, "峰峰值": false, "整流平均值": false, "均方根": false, "方根幅值": false, "波形因子": false, "峰值因子": false, "脉冲因子": false, "裕度因子": false, "重心频率": false, "均方频率": false, "均方根频率": false, "频率方差": false, "频率标准差": false, "谱峭度的均值": false, "谱峭度的峰度": false
                    }
                }, "optional": true
            },
            {
                "label_display": "单传感器信息重要性特征选择", "label": "特征选择", "id": "1.3", "nodeId": "1.3", "nodeContainerStyle": { "left": "502px", "top": "50px" }, "use_algorithm": "mutual_information_importance", "parameters": {
                    "feature_imp": { "rule": 1, "threshold1": null, "threshold2": null }, "mutual_information_importance": { "rule": 1, "threshold1": 0.5, "threshold2": null },
                    "correlation_coefficient_importance": { "rule": 1, "threshold": 0.005 }, "feature_imp_multiple": { "rule": 1, "threshold": 0.005 }, "mutual_information_importance_multiple": { "rule": 1, "threshold": 0.005 }, "correlation_coefficient_importance_multiple": { "rule": 1, "threshold": 0.005 }
                }, "optional": true
            },
            { "label_display": "单传感器随机森林故障诊断", "label": "故障诊断", "id": "2.1", "nodeId": "2.1", "nodeContainerStyle": { "left": "726px", "top": "50px" }, "use_algorithm": "random_forest", "parameters": { "random_forest": {}, "svc": {}, "gru": {}, "lstm": {}, "random_forest_multiple": {}, "svc_multiple": {}, "gru_multiple": {}, "lstm_multiple": {}, "ulcnn": {}, "ulcnn_multiple": {}, "spectrumModel": {}, "spectrumModel_multiple": {} }, "optional": false },
            { "label_display": "单传感器层次分析模糊综合评估法", "label": "层次分析模糊综合评估", "id": "3.1", "nodeId": "3.1", "nodeContainerStyle": { "left": "980px", "top": "50px" }, "use_algorithm": "FAHP", "parameters": { "FAHP": {}, "FAHP_multiple": {} }, "optional": false }], "connection": ["数据源", "特征提取", "特征选择", "故障诊断", "层次分析模糊综合评估"]
        }
    }
};



// 算法简介
export const plainIntroduction = {
    polynomial_interpolation:
        "# 多项式插值方法\n" +
        "## 多项式插值是一种数学技术，通过构造一个多项式来精确地通过一组给定的数据点，从而对数据进行平滑逼近。\n" +
        "# 使用场景\n" +
        "### 1. **平滑逼近**：适用于需要对数据点进行精确插值的情况，以发现数据之间的潜在趋势。\n" +
        "### 2. **曲线拟合**：在数据点之间建立一个连续的多项式曲线，用于表示数据的整体形态。\n" +
        "### 3. **信号重建**：在信号采集中，用于填补缺失的数据点，恢复信号的完整性。\n" +
        "### 4. **滤波和去噪**：在信号处理中，多项式插值可以帮助平滑信号，减少噪声的影响。",
    neighboring_values_interpolation:
        "# 邻近值插补在数据预处理中的应用\n" +
        "## 邻近值插补是一种基于数据点之间相似性或距离的数据插补方法，用于处理缺失数据。它通过选择缺失值附近已有的值来填补空缺。\n" +
        "# 插补的使用场景\n" +
        "### 1. 时间序列数据：在时间序列分析中，当数据集中出现缺失值时，可以使用邻近时间点的值进行插补。\n" +
        "### 2. 快速预处理：在需要快速进行数据预处理，而没有时间或资源进行更复杂的插补方法时，邻近值插补是一种实用的选择。\n" +
        "### 3. 异常值处理：当缺失值可能是由于数据收集过程中的异常或错误造成时，邻近值插补可以作为一种简单的错误校正方法。\n" +
        "### 4. 数据预处理：在数据预处理阶段，邻近值插补可以用于填补由于数据录入错误或丢失造成的缺失值。\n" +
        "### 5. 缺失数据不多的情况：当数据集中的缺失值不多，且分布均匀时，邻近值插补提供了一种快速且有效的解决方案。",
    bicubic_interpolation:
        "# 双三次插值方法\n" +
        "## 双三次插值是一种高效的插值技术，它通过构造一个在数据点及其一阶和二阶导数上都连续的三次多项式来逼近数据。\n" +
        "# 使用场景\n" +
        "### 1. **高精度逼近**：适用于需要在数据点之间进行平滑且高精度插值的情况。\n" +
        "### 2. **曲面建模**：在二维数据集中，用于创建连续的曲面模型，以便于分析和可视化。\n" +
        "### 3. **高质量信号重建**：在信号采集中，用于填补缺失的数据点，同时保持信号的平滑性和连续性。\n" +
        "### 4. **图像处理**：在图像缩放和旋转中，双三次插值可以减少锯齿效应，保持图像质量。\n",
    lagrange_interpolation:
        "# 拉格朗日插值法\n" +
        "## 拉格朗日插值法是一种多项式插值方法，通过构造一个多项式来精确匹配一组给定的数据点。\n" +
        "# 使用场景\n" +
        "### 1. **精确逼近**：适用于需要在数据点之间进行精确插值的情况，尤其是在数据点数量较少时。\n" +
        "### 2. **曲线拟合**：在数据点之间建立一个多项式曲线，用于模拟数据的整体趋势。\n" +
        "### 3. **信号重建**：在信号采集中，用于填补丢失的数据点，恢复信号的完整性。\n" +
        "### 4. **数据平滑**：在信号处理中，拉格朗日插值可以帮助平滑信号，减少噪声的影响。\n",
    newton_interpolation:
        "# 牛顿插值法\n" +
        "## 牛顿插值法是一种高效的多项式插值技术，它利用差商构建一个多项式，能够通过一组给定的数据点。\n" +
        "# 使用场景\n" +
        "### 1. **递归构建**：适用于需要逐步增加数据点进行插值的情况，便于更新和维护多项式模型。\n" +
        "### 2. **曲线拟合**：在数据点之间构建一个多项式曲线，用于模拟数据的趋势和模式。\n" +
        "### 3. **动态信号重建**：在信号处理中，牛顿插值可以动态地填补丢失的数据点，适应信号变化。\n" +
        "### 4. **实时数据处理**：适用于需要实时处理和更新数据的场合，如在线信号分析。\n",
    linear_interpolation:
        "# 线性插值方法\n" +
        "## 线性插值是一种基本的插值方法，通过在两个已知数据点之间构建一条直线来估计未知数据点的值。\n" +
        "# 使用场景\n" +
        "### 1. **简单估算**：在需要快速且直接的数据点估计时使用，适用于数据变化趋势为线性的情况。\n" +
        "### 2. **趋势分析**：用于识别和展示数据的线性关系，便于理解数据的基本情况。\n" +
        "### 3. **数据填补**：在信号采集中，用于填补因测量误差或数据丢失造成的空白。\n" +
        "### 4. **去噪处理**：在信号处理中，线性插值可以用于简化信号，减少高频噪声。\n",
    time_domain_features:
        "# 时域特征提取\n" +
        "## 时域特征反映了信号在时间维度上的特性，不涉及任何频率转换，如傅里叶变换。而时域特征提取是从信号的原始时间序列数据中抽取关键信息的过程，这些信息能够表征信号的基本属性和内在特性。\n" +
        "# 主要方法\n" +
        "### 1. **峰值检测**：识别信号的最大或最小值点。\n" +
        "### 2. **统计分析**：计算信号的均值、方差、偏度、峭度等统计量。\n" +
        "### 3. **能量计算**：评估信号的总能量，通常通过对信号平方后积分。\n" +
        "### 4. **时间参数测量**：测量信号的周期、持续时间、延迟等。\n" +
        "### 5. **波形特征分析**：提取波形的特定形状特征，如脉冲宽度、上升时间等。\n" +
        "### 6. **相关性分析**：计算信号与参考信号之间的相关度。\n" +
        "### 7. **自相关函数**：分析信号在不同时间延迟下的相关性。\n",
    frequency_domain_features:
        "# 频域特征提取\n" +
        "## 频域特征提取是一种分析技术，它通过将信号从时域转换到频域来提取信号的频率成分，进而分析和处理信号。\n" +
        "# 主要方法\n" +
        "### 1. **傅里叶变换(FT)**：将时域信号转换为频域表示。\n" +
        "### 2. **短时傅里叶变换(STFT)**：分析时变信号的局部频率特性。\n" +
        "### 3. **小波变换(WT)**：提供时间和频率的局部化信息，适用于非平稳信号分析。\n" +
        "### 4. **谱估计技术**：如周期图法、协方差法等，用于更精细的频谱分析。\n",
    time_frequency_domain_features:
        "# 时频域特征提取\n" +
        "## 时频域特征提取结合了时域和频域的方法来研究信号的局部特性，时频域分析不仅关注信号在单一时间点的特征，也关注信号在不同时间段的频率变化，适用于分析非平稳信号。\n" +
        "# 主要方法\n" +
        "### 1. **短时傅里叶变换(STFT)**：通过在不同时间窗口上应用傅里叶变换来分析信号的局部频率特性。\n" +
        "### 2. **小波变换(WT)**：利用小波函数来分析信号在不同时间和频率尺度上的特性。\n" +
        "### 3. **Wigner-Ville分布**：一种二维时频表示，能够展示信号的频率和时变特性。\n" +
        "### 4. **Hilbert-Huang变换(HHT)**：结合经验模态分解(EMD)和Hilbert变换，用于分析非线性和非平稳信号。\n",

    FAHP:
        "# 层次分析模糊综合评估法\n" +
        "## 层次分析模糊综合评估法是一种将模糊综合评价法和层次分析法相结合的评价方法，在体系评价、效能评估，系统优化等方面有着广泛的应用，是一种定性与定量相结合的评价模型，一般是先用层析分析法确定因素集，然后用模糊综合评判确定评判效果。模糊法是在层次法之上，两者相互融合，对评价有着很好的可靠性。\n" +
        "# 评价过程\n" +
        "### 1. **建立层次结构模型**：首先使用层次分析法确定问题的目标层、准则层和方案层。\n" +
        "### 2. **成对比较和一致性检验**：通过成对比较确定各因素的相对重要性，并进行一致性检验。\n" +
        "### 3. **确定权重向量**：计算准则层和方案层的权重向量。\n" +
        "### 4. **构建模糊评价模型**：利用模糊综合评价法构建评价模型，确定评价指标的隶属度函数。\n" +
        "### 5. **模糊综合评判**：综合考虑各因素的权重和隶属度，进行模糊综合评判，得出最终的评价结果。\n" +
        "\n",
    BHM:
        "# 层次朴素贝叶斯评估法\n" +
        "## 层次朴素贝叶斯评估法是一种将层次分析和高斯朴素贝叶斯相结合的评价方法，在体系评价、效能评估，系统优化等方面有着广泛的应用，是一种定性与定量相结合的评价模型，一般是先用层析分析法确定因素集，然后用高斯朴素贝叶斯方法建立特征的高斯分布，并最终确定评判效果。\n" +
        "# 算法特点\n" +
        "### 1. 可以综合考虑多个因素的影响，给出全面评价结果\n" +
        "### 2. 对给定的问题能够提供概率形式的决策支持 \n" +
        "### 3. 能够处理多类分类问题，并且对缺失数据和不完整数据的处理能力较强",
    BHM_multiple:
        "# 层次朴素贝叶斯评估法\n" +
        "## 层次朴素贝叶斯评估法是一种将层次分析和高斯朴素贝叶斯相结合的评价方法，在体系评价、效能评估，系统优化等方面有着广泛的应用，是一种定性与定量相结合的评价模型，一般是先用层析分析法确定因素集，然后用高斯朴素贝叶斯方法建立特征的高斯分布，并最终确定评判效果。\n" +
        "# 算法特点\n" +
        "### 1. 可以综合考虑多个因素的影响，给出全面评价结果\n" +
        "### 2. 对给定的问题能够提供概率形式的决策支持 \n" +
        "### 3. 能够处理多类分类问题，并且对缺失数据和不完整数据的处理能力较强",
    FAHP_multiple:
        "# 层次分析模糊综合评估法\n" +
        "## 层次分析模糊综合评估法是一种将模糊综合评价法和层次分析法相结合的评价方法，在体系评价、效能评估，系统优化等方面有着广泛的应用，是一种定性与定量相结合的评价模型，一般是先用层析分析法确定因素集，然后用模糊综合评判确定评判效果。模糊法是在层次法之上，两者相互融合，对评价有着很好的可靠性。\n" +
        "# 评价过程\n" +
        "### 1. **建立层次结构模型**：首先使用层次分析法确定问题的目标层、准则层和方案层。\n" +
        "### 2. **成对比较和一致性检验**：通过成对比较确定各因素的相对重要性，并进行一致性检验。\n" +
        "### 3. **确定权重向量**：计算准则层和方案层的权重向量。\n" +
        "### 4. **构建模糊评价模型**：利用模糊综合评价法构建评价模型，确定评价指标的隶属度函数。\n" +
        "### 5. **模糊综合评判**：综合考虑各因素的权重和隶属度，进行模糊综合评判，得出最终的评价结果。\n" +
        "\n",
    feature_imp:
        "# 使用树模型进行特征选择\n" +
        "\n" +
        "## 特征选择是机器学习中的一项关键任务，用于识别最有信息量的特征，以提高模型的性能和可解释性。树模型，包括决策树、随机森林和梯度提升树等，提供了多种特征选择的方法。特征选择旨在从原始特征集中挑选出对模型预测最有用的特征子集。在树模型中，这一过程通常基于以下原理：\n" +
        "\n" +
        "### 1. **分裂准则**：树模型在分裂节点时选择特征，基于如信息增益、基尼不纯度等准则。\n" +
        "### 2. **特征重要性**：衡量特征在模型中的贡献度，对模型性能的影响.\n" +
        "### 3. **数据驱动**：特征选择过程完全基于数据和模型的反馈，而不是基于领域知识或其他外部标准。",
    mutual_information_importance:
        "# 互信息重要性特征选择\n" +
        "## 互信息是度量两个随机变量之间相互依赖性的统计量，它在特征选择中用于识别最有信息量的特征。互信息重要性特征选择遵循以下原理：\n" +
        "### 1. 最大化信息量：选择那些包含关于目标变量最多信息的特征。\n" +
        "### 2. 减少冗余： 避免选择相互之间提供重复信息的特征。\n" +
        "### 3. 计算效率: 采用有效的算法来计算互信息，以保证特征选择过程的可行性。\n" +
        "### 4. 稳健性; 确保特征选择方法对于数据的小变化是稳健的。\n" +
        "### 5. 适应性： 特征选择方法应能适应不同类型的数据分布。\n" +
        "### 6. 降维： 通过特征选择减少特征空间的维度，以简化模型并提高计算效率。\n" +
        "### 7. 模型无关：互信息特征选择是模型无关的，即它不依赖于特定的预测模型。\n" +
        "### 8. 增量学习：在新数据到来时，能够更新特征选择结果，适应数据的变化。\n",
    correlation_coefficient_importance:
        "# 相关系数重要性特征选择\n" +
        "## 相关系数特征选择是一种评估特征与目标变量之间线性关系强度的方法。其中相关系数是量化两个变量之间线性关系的统计度量，通常使用皮尔逊相关系数。其遵循以下原理：\n" +
        "### 1. 线性关系：假设特征与目标变量之间存在线性关系，通过相关系数评估这种关系的强度。\n" +
        "### 2. 特征有效性：选择那些与目标变量具有显著线性相关性的特征，以提高模型的预测能力。\n" +
        "### 3. 多重共线性控制： 在选择特征时，避免纳入高度线性相关的特征，减少模型的多重共线性问题。\n" +
        "### 4. 简洁性原则：倾向于选择较少的特征，以简化模型结构，提高模型的可解释性和泛化能力。\n" +
        "### 5. 计算效率： 相关系数的计算相对简单快速，适用于大规模数据集的特征选择。\n" +
        "### 6. 稳健性：考虑特征选择方法对于数据中的异常值和噪声的敏感性，确保选择结果的稳定性。\n" +
        "### 7. 模型无关性：虽然基于线性关系选择特征，但所选特征适用于不同类型的预测模型。\n" +
        "### 8. 适应性：特征选择方法应能够适应不同的数据特征和分布情况，保持选择结果的准确性。",
    random_forest:
        "# 随机森林故障诊断\n" +
        "## 随机森林是一种集成学习方法，广泛应用于故障诊断，能够处理复杂的数据模式和识别多种故障类型。其中随机森林是由多个决策树构成的集成模型，每棵树在数据集的不同子集上训练，并对结果进行投票或平均以得出最终预测。\n" +
        "# 特点\n" +
        "### 1.高准确性：集成多个决策树的预测结果，提高整体的诊断准确性。\n" +
        "### 2.鲁棒性：对数据中的噪声和异常值具有较好的抵抗力。\n" +
        "### 3.多故障类型识别：能够同时处理和识别多种不同类型的故障。\n" +
        "### 4.捕捉非线性关系;有效识别数据中的非线性故障模式和复杂关系。\n" +
        "### 5.模型泛化能力:由于集成了多个树，随机森林具有良好的泛化能力，减少过拟合风险。\n",
    svc:
        "# 支持向量机（SVM）故障诊断\n" +
        "## 支持向量机是一种在故障诊断中广泛使用的监督学习模型，以其在分类和模式识别任务中的强大性能而著称。支持向量机是一种基于间隔最大化原则来构建分类器的方法，特别适用于高维数据和非线性问题。\n" +
        "# 特点\n" +
        "### 1.高维数据处理能力:SVM通过核技巧有效地处理高维数据，无需显式地映射到高维空间。\n" +
        "### 2.间隔最大化:SVM通过最大化数据点之间的间隔来提高分类的鲁棒性。\n" +
        "### 3.核函数:使用不同的核函数（如线性核、多项式核、径向基函数核等）来处理线性不可分的数据。\n" +
        "### 4.软间隔引入:允许一些数据点违反间隔，以提高模型的泛化能力。\n" +
        "### 5.正则化:通过正则化项控制模型复杂度，防止过拟合。\n" +
        "### 6.多类分类:通过策略如一对多（OvR）方法，SVM能够处理多类故障诊断问题。\n",
    gru:
        "# 门控循环单元（GRU）故障诊断\n" +
        "## 门控循环单元是一种特殊的循环神经网络，适用于序列预测和时间序列分析的递归神经网络结构，它引入了更新门和重置门机制，以改善梯度流动并捕捉长期依赖关系，特别适用于故障诊断任务。\n" +
        "# 特点\n" +
        "### 1. 长期依赖学习能力：GRU特别设计了更新门来解决传统RNN中的梯度消失问题，使其能够学习长期依赖信息。\n" +
        "### 2. 动态时间序列处理能力:GRU能够处理时间序列数据中的动态变化，适用于捕捉故障发生前后的模式变化。\n" +
        "### 3. 门控机制的灵活性:通过更新门和重置门的控制，GRU可以灵活地决定信息的保留和遗忘，以适应不同的故障特征。\n" +
        "### 4. 易于集成和训练:GRU模型易于在现有的深度学习框架中实现和训练，便于与其他模型或数据处理流程集成。\n",
    lstm:
        "# 长短期记忆网络（LSTM）故障诊断\n" +
        "## 长短期记忆网络是一种特殊类型的循环神经网络（RNN），设计用来解决传统RNN在处理长序列数据时的梯度消失问题。LSTM因其出色的记忆和遗忘机制，在序列预测和时间序列分析中表现卓越，非常适合故障诊断任务。\n" +
        "# 特点\n" +
        "### 1. 有效的长期依赖处理:LSTM通过其复杂的门控机制（输入门、遗忘门、输出门）来控制信息的流动，有效捕捉和记忆长期依赖关系。\n" +
        "### 2. 强大的序列预测能力:LSTM能够分析时间序列数据中的复杂模式，预测故障发生的概率和时间点。\n" +
        "### 3. 适应性强的门控机制:通过遗忘门和输入门的协同工作，LSTM可以决定哪些信息应该被遗忘，哪些信息应该被更新和保留。\n" +
        "### 4. 良好的泛化能力:经过适当的训练，LSTM可以学习到数据中的深层特征，对未见过的故障模式具有良好的泛化和识别能力。\n",
    ulcnn:
        "# 一维卷积神经网络故障诊断\n" +
        "## 一维卷积神经网络基于卷积运算，通过在输入数据上滑动一个固定大小的卷积核来提取局部特征。这些特征随后通过池化层进行降采样，以减少数据的空间大小并提取更高级别的特征。最终，通过全连接层进行分类、回归或其他任务。\n" +
        "# 特点\n" +
        "### 1. 特征提取:一维卷积神经网络能够提取局部特征，对时间序列数据中的局部模式进行学习。\n" +
        "### 2. 降维:通过卷积操作，一维卷积神经网络能够对高维数据进行降维，降低计算复杂度。\n" +
        "### 3. 模型复杂度:由于一维卷积神经网络具有复杂的结构，其模型复杂度较高，但可以处理更长的序列数据。\n",
    spectrumModel:
        "# 时频图卷积模型的故障诊断\n" +
        "## 时频图卷积模型是一种用于故障诊断的深度学习模型，它利用时频图（Spectrogram）对时间序列数据进行特征提取，并通过卷积神经网络进行分类。时频图是一种将时间序列数据分解为频率和时间轴上的二维图像的方法，可以捕获时间序列中的周期性特征和频率信息。\n" +
        "# 特点\n" +
        "### 1. 特征提取:时频图卷积模型能够提取时间序列数据中的周期性特征和频率信息，对故障诊断任务具有很好的特征提取能力。\n" +
        "### 2. 降噪能力:时频图卷积模型能够对噪声数据进行降噪，对故障诊断任务具有很好的鲁棒性。\n" +
        "### 3. 鲁棒性强:由于采用了卷积和池化操作，该模型对输入数据的微小变化具有一定的鲁棒性。",
    wavelet_trans_denoise:
        "# 小波变换去噪\n" +
        "## 小波变换去噪是一种利用小波分析对信号进行降噪处理的方法，它通过将信号分解为不同时间尺度上的成分，然后有选择地去除噪声成分。\n" +
        "# 基本原理\n" +
        "### 1. 多尺度分解：小波变换将信号分解为不同时间尺度（或频率）上的成分，这些成分称为小波系数。\n" +
        "### 2. 信号与噪声的分：信号往往在小波变换的低频部分具有较大的系数，而噪声则在高频部分较为显著。\n" +
        "### 3. 阈值处理：对小波系数进行阈值处理，设置一个阈值，将小于该阈值的系数视为噪声并置零或进行缩减，而保留较大的系数。\n" +
        "### 4. 重构信号:通过保留和放大重要的小波系数，忽略或减弱噪声成分，然后通过小波逆变换重构出降噪后的信号。\n",
    max_min:
        "# 最大最小值归一化（Max-Min Normalization）\n" +
        "## 最大最小值归一化通过一个简单的线性变换过程，将数据特征的值缩放到[0, 1]的范围内。这个过程包括识别数据集中每个特征的最大值和最小值，然后利用这两个值来调整所有数据点，确保最小的数据点映射到0，最大的数据点映射到1，而其他点则根据它们与最小值和最大值的关系被映射到(0, 1)区间内。如果需要不同的数值范围，可以通过额外的缩放和平移操作来实现。这种方法易于实现且计算效率高，但要注意它对数据中的极端值或异常值较为敏感。\n" +
        "# 特点\n" +
        "### 简单性：最大最小值归一化方法简单，易于实现。\n" +
        "### 快速性：计算过程快速，适合大规模数据集。\n" +
        "### 数据分布敏感：归一化结果依赖于数据中的最小值和最大值，对异常值敏感。",
    "z-score":
        "# z-score标准化\n" +
        "## z-score标准化是一种数据预处理技术，用于将数据转换为具有平均值为0和标准差为1的标准分数。这种转换基于原始数据的均值和标准差，使得转换后的数据分布更加规范化，便于比较和分析。\n" +
        "# 特点\n" +
        "### 1. 中心化和尺度统一：数据通过减去均值并除以标准差进行转换，结果是一个中心化在0，单位标准差的分布。\n" +
        "### 2. 正态分布适配：该方法假设数据近似正态分布，通过转换使得数据更接近标准正态分布。\n" +
        "### 3. 异常值敏感性较低：与基于极端值的方法不同，z-score标准化使用均值和标准差，因此对异常值的影响较小。\n" +
        "### 4. 易于解释性：转换后的z-score值表示数据点距离均值的标准差数，提供了数据点分布情况的直观度量。\n",
    robust_scaler:
        "# 鲁棒标准化\n" +
        "## 鲁棒标准化是一种数据预处理技术，它使用数据的中位数和四分位数范围（IQR）来缩放数据，从而对异常值具有较高的抵抗力。这种方法不依赖于数据的均值和标准差，而是使用中位数和IQR来确定数据的尺度。\n" +
        "# 特点\n" +
        "### 1. 对异常值的鲁棒性：鲁棒标准化通过使用中位数和IQR，减少了异常值对数据缩放的影响。\n" +
        "### 2. 中位数和四分位数：数据的中心位置由中位数确定，而数据的尺度由IQR（即第三四分位数与第一四分位数之差）确定。\n" +
        "### 3. 缩放方法：鲁棒标准化通常涉及将数据点减去中位数，然后除以IQR的一定比例（通常是1/0.7413，这个值使得IQR近似等于标准差的1倍）。\n",
    max_abs_scaler:
        "# 最大绝对值标准化\n" +
        "## 最大绝对值标准化是一种数据预处理技术，通过将数据的每个特征的值除以其最大绝对值来实现标准化。这种方法不关心数据的正负符号，只关注值的大小。\n" +
        "# 特点\n" +
        "### 1. 简单易行：最大绝对值标准化的计算过程简单，易于实现和理解。\n" +
        "### 2. 忽略数据符号：该方法只考虑数据的绝对值，因此对数据的正负符号不敏感。\n" +
        "### 3. 抵抗异常值：由于只依赖于最大绝对值，该方法对异常值具有一定的抵抗力。\n" +
        "### 4. 缩放到[-1, 1]区间：标准化后的数据将位于[-1, 1]区间内，便于比较不同特征的值。",
    linear_regression:
        "# 线性回在趋势预测\n" +
        "## 线性回归是一种用于确定因变量（预测目标）与一个或多个自变量（预测因子）之间线性关系的方法。在趋势预测中，线性回归模型通过拟合历史数据来预测未来的趋势或模式。\n" +
        "# 特点\n" +
        "### 1. 模型表达性：线性回归模型通过直线（单变量线性回归）或平面/超平面（多元线性回归）来表达数据之间的关系。\n" +
        "### 2. 预测连续变量：主要用于预测连续变量，如房价、气温、销售额等。\n" +
        "### 3. 结合定性变量：通过哑变量（Dummy Variables）可以包含定性变量，以研究它们对趋势的影响。\n" +
        "### 4. 基于最小二乘法：通常使用最小二乘法来估计模型参数，这种方法可以找到最佳拟合直线或超平面。\n" +
        "### 5. 假设正态分布：在简单线性回归中，假设误差项呈正态分布，且具有常数方差。\n" +
        "### 6. 稳健性分析：虽然对异常值敏感，但通过残差分析和杠杆值可以识别并处理异常值和高影响力点。\n" +
        "### 7. 可扩展性：从简单的单变量线性回归可以扩展到包含多个预测变量的多元线性回归。\n",
    deeplearning_interpolation:
        "# 基于LSTM网络的深度学习插值算法\n" +
        "## 基于LSTM的深度学习插补，是通过使用已标记（即非缺失）的训练数据对LSTM网络进行训练，然后使用训练好的LSTM模型对测试集（包含缺失值的数据）进行插值预测。\n" +
        "## 特点\n" +
        "### 1. 强大的序列建模能力：LSTM模型能够捕捉时间序列中的长期依赖关系，从而更准确地预测缺失值\n" +
        "### 2. 自适应性：通过深度学习的自动特征提取能力，LSTM模型能够从原始数据中学习到复杂的非线性关系，无需进行复杂的手工特征提取。\n"
    ,

    // '### 8. 包含交互作用和多项式项：可以通过添加交互作用项和多项式项来提高模型的预测能力。\n',
    随机森林趋势预测:
        "# 随机森在趋势预测\n" +
        "## 随机森林是一种集成学习方法，它通过构建多个决策树并将它们的预测结果进行汇总，以提高模型的准确性和鲁棒性。在趋势预测中，随机森林能够捕捉数据中的复杂模式和非线性关系。\n" +
        "# 特点\n" +
        "### 1. 高准确性：随机森林通过集成多个决策树的预测结果，降低了模型的方差，提高了预测的准确性。\n" +
        "### 2. 自动特征选择：随机森林在构建决策树的过程中，可以评估特征的重要性，从而实现自动特征选择。\n" +
        "### 3. 强大的非线性拟合能力：随机森林能够处理高度复杂的数据模式，适用于非线性趋势的预测。\n" +
        "### 4. 鲁棒性：由于集成了多个决策树，随机森林对异常值和噪声具有较强的鲁棒性。\n" +
        "### 5. 易于实现和并行化：随机森林模型易于实现，并且其训练过程可以并行化，提高了计算效率。\n" +
        "### 6. 多变量处理能力：随机森林能够同时处理多个变量，捕捉它们与预测目标之间的复杂关系。\n",
    SVM的趋势预测:
        "# 支持向量机（SVM）趋势预测\n" +
        "## 支持向量机是一种监督学习模型，用于分类和回归任务。在趋势预测中，SVM通过找到数据中的最优边界或超平面，对数据的未来趋势进行预测。\n" +
        "# 特点\n" +
        "### 1. 优秀的泛化能力：SVM通过选择支持向量来构建模型，这些向量定义了决策边界，使得模型具有很好的泛化能力。\n" +
        "### 2. 核技巧：SVM使用核函数将数据映射到高维空间，以处理非线性趋势预测问题。\n" +
        "### 3. 正则化控制：通过正则化参数控制模型的复杂度，避免过拟合，确保模型的稳定性。\n" +
        "### 4. 适用于小样本数据：SVM在小样本情况下也能表现出较好的预测性能，适合样本量不足的趋势预测任务。\n" +
        "### 5. 模型解释性：相比于一些黑盒模型，SVM具有一定的解释性，特别是通过支持向量可以了解模型决策的关键数据点。\n" +
        "### 6. 多类趋势预测：通过适当的策略，SVM可以扩展到多类趋势预测问题。\n",
    GRU的趋势预测:
        "# 门控循环单元（GRU）趋势预测\n" +
        "## 门控循环单元是一种特殊类型的递归神经网络（RNN），设计用于处理序列数据，能够捕捉时间序列中的动态特征和长期依赖关系。在趋势预测中，GRU能够学习数据随时间变化的模式，并预测未来的趋势。\n" +
        "# 特点\n" +
        "### 1. 捕捉时间序列特性：GRU通过其门控机制能够捕捉时间序列数据中的短期和长期依赖关系，适用于具有时间连贯性的趋势预测。\n" +
        "### 2. 门控调节信息流：利用更新门和重置门，GRU能够有选择地更新或保留状态信息，从而适应不同时间尺度的趋势变化。\n" +
        "### 3. 处理非线性动态：GRU能够处理复杂的非线性时间序列数据，对于预测不规则或周期性变化的趋势特别有效。\n" +
        "### 4. 避免梯度消失问题：GRU的设计有助于缓解传统RNN中的梯度消失问题，使得网络能够学习长期时间依赖。\n" +
        "### 5. 易于集成和训练：现代深度学习框架提供了GRU的实现，易于集成到趋势预测模型中，并支持大规模数据集的训练。\n" +
        "### 6. 多步时间预测：GRU不仅可以进行单步预测，还可以通过序列到序列模型进行多步时间预测。\n",
    LSTM的趋势预测:
        "# 长短期记忆网络（LSTM）在趋势预测中的应用\n" +
        "## 长短期记忆网络是一种高级的递归神经网络（RNN），专为解决传统RNN在处理长序列数据时遇到的梯度消失或梯度爆炸问题而设计。LSTM在趋势预测中能够学习时间序列数据中的长期依赖关系，并进行有效的预测。\n" +
        "# 特点\n" +
        "### 1. 长期依赖学习：LSTM的特别设计使其能够捕捉时间序列中相隔很远的依赖关系，这对于理解长期趋势至关重要。\n" +
        "### 2. 门控机制：过其复杂的门控机制（输入门、遗忘门、输出门），LSTM能够决定信息的保留和遗忘，从而适应时间序列数据的变化。\n" +
        "### 3. 处理非线性：LSTM能够处理时间序列数据中的非线性模式，适用于预测复杂的趋势变化。\n" +
        "### 4. 避免过拟合：由于其门控单元的结构，LSTM在训练过程中更不容易过拟合，提高了模型的泛化能力。\n" +
        "### 5. 多步预测能力：LSTM可以设计为序列到序列模型，进行多步趋势预测，而不仅仅是单步预测。\n",
    添加噪声:
        "### 添加噪声算法可对输入信号添加如高斯噪声等常见的噪声\n***\n ### 添加噪声的主要应用有：\n **1. 增强信号检测。在某些特定的非线性系统中，噪声的存在能够增强微弱信号的检测能力，这种现象被称为随机共振。**\n \
    **2. 减小重构误差。如果在信号处理中只加入正的白噪声，那么在重构过程中可能会多出来加入的白噪声，从而增大了重构误差。因此，加入正负序列的白噪声可以在分解过程中相互抵消，减小重构误差。**\n### 本模块包含算法如下:\n **高斯白噪声： 高斯白噪声(White Gaussian Noise)在通信、信号处理及科学研究等\
    多个领域中扮演着重要角色。它作为一种理想的噪声模型，具有独特的统计特性和功率谱分布，为系统性能评估、算法测试及信号分析提供了有力工具**",
    插值处理:
        "### 插值处理算法可以对输入信号进行插值操作\n***\n ### 插值处理的主要应用有：\n **1. 数值计算。函数逼近：插值方法，如拉格朗日插值多项式，可以用于在给定节点上逼近任意函数，从而在节点外的位置计算函数的近似值。**\n \
    **2. 图像处理。图像放大与缩小：双线性插值等方法可以用于图像的放大或缩小，以获取更高分辨率或适当尺寸的图像。图像平滑：线性插值等方法可以用于平滑图像中的噪声或异常值，提高图像质量。**\
    **3. 数据净化。缺失值处理：插值法是一种有效的缺失值处理方法，能够根据不同情况选择合适的插值类型进行估算，实现数据的完整性和连续性。数据平滑：插值法可以用于构建连续且平滑的函数模型来拟合数据，消除噪声和异常值的影响，提高数据质量。**\n \
    ### 本插值处理模块中包含算法：\n **多项式插值算法、双三次插值算法、拉格朗日插值算法、牛顿插值算法、线性插值算法**",
    TDF:
        "### 特征提取算法可对输入信号进行人工特征提取\n***\n ### 特征提取算法中主要包括信号时域特征和频域特征的提取：\n **1. 时域特征。定义：时域特征描述的是信号随时间变化的关系。在时域中，信号被表示为时间的函数，其动态信号描述信号在不同时刻的取值。\
    特点：时域表示较为形象与直观，对于正弦信号等简单波形的时域表达，可以通过幅值、频率、相位三个基本特征值唯一确定。时域分析能够直接反映信号随时间的实时变化。**\n **2. 频域特征。定义：频域特征描述的是信号在频率方面的特性。\
    频域分析是通过对信号进行傅立叶变换等数学方法，将信号从时间域转换到频率域，从而研究信号的频率结构。特点：频域分析能够深入剖析信号的本质，揭示信号中不易被时域分析发现的特征。频域特征通常用于表达信号的周期性信息。**\n ### 本特征提取算法中提取的时域和频域的特征包括：\
    \n **1. 时域特征：均值、方差、标准差、峰度、偏度、四阶累积量、六阶累积量、最大值、最小值、中位数、峰峰值、整流平均值、均方根、方根幅度、波形因子、峰值因子、脉冲因子、裕度因子**\n \
    **1. 频域特征：重心频率、均方频率、均方根频率、频率方差、频率标准差、谱峭度的均值、谱峭度的标准差、谱峭度的峰度、谱峭度的偏度**",
    层次分析模糊综合评估:
        "### 层次分析模糊综合评估法是一种将模糊综合评价法和层次分析法相结合的评价方法\n***\n ### 算法优点：\n **1. 可以综合考虑多个因素的影响，给出全面评价结果。**\n \
    **2. 评价结果是一个矢量，而不是一个点值，包含的信息比较丰富，既可以比较准确的刻画被评价对象，又可以进一步加工，得到参考信息。**\n **3. 模糊评价通过精确的数字手段处理模糊的评价对象，能对蕴藏信息呈现模糊性的资料作出比较科学、合理、贴近实际的量化评价。**",

};

// 算法参数名称
export const labelsForParams = {
    SNR: '信噪比',
    layers: '网络层数',
    num_workers: '工作线程数',
    num_features: '选取特征数量',
    wavelet: '小波类型',
    wavelet_level: '小波层数',
    useLog: '使用训练模型时的标准化方法'
}