程序框架
1、计算晴空辐照度
   def clear_sky_model(latitude, lontitude)
2、计算晴空指数
   def clear_date_index_new(power, cap)
   输入数据要整天
3、历史数据拟合晴空功率模型
   def fit_model(power, E, nwp, set_2017):
       return P_final_single, P_altitude, ratio_test, K_Tc