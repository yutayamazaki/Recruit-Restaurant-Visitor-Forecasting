# Recruit-Restaurant-Visitor-Forecasting
## How to Use

```shell
git clone https://github.com/yutayamazaki/Recruit-Restaurant-Visitor-Forecasting
cd Recruit-Restaurant-Visitor-Forecasting
pipenv shell
# python train.py
# or
# cd notebooks
# jupyter lab
```

### Overview
リクルートによるレストラン来客数予測コンペ  
評価指標はRMSLEであり，以下の式で計算される(詳細や実装は[notebooks/EDA.ipynb](https://github.com/yutayamazaki/Recruit-Restaurant-Visitor-Forecasting/blob/master/notebooks/EDA.ipynb)に)  

<img src="https://latex.codecogs.com/gif.latex?RMSLE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(log(p_i+1)-log(a_i+1))^2}"/>

意図するところとしては，一般に回帰で用いられるRMSEと比較し，
- logを取っているので，予測が実際を下回る場合よりも，予測が実際を上回る場合によりペナルティを課す事になる
- これはレストランにおいて，予測が実際を上回ると，材料を切らすこととなり，来客に対する売り上げが減少するからと考えられる．逆に，予測が実際を下回っても，少し在庫を抱えるだけで済むので，利益の減少には直結しない  
- 客数の分布がかなり偏っているので，正規分布に近づけたいという意図もありそう  


## Data Description
### air_reserve.csv
AirREGIというリクルートが提供するPOSレジアプリのデータ  
- air_store_id: AirREGIにおける，各レストランのid
- visit_datetime: 予約時間
- reserve_datetime: 予約された時間
- reserve_visitors: 各予約での来客数

### air_store_info.csv
- air_store_id: AirREGIにおける，各レストランのid
- air_genre_name: AirREGIに登録されている店のジャンル(和食，イタリアンなど)
- air_area_name: 都道府県・市町村の情報
- latitude, longitude: 緯度・経度

### air_visit_data.csv
- air_store_id: AirREGIにおける，各レストランのid
- visit_date: 日付
- visitors: その日，そのレストランの来客数

### date_info.csv
- calendar_date: 日付
- day_of_week: 曜日
- holiday_flg: 祝日を表すbinary(土日は含まれない)

### hpg_reserve.csv
- hpg_store_id: HotPepperGourmetの各レストランのid
- visit_datetime: 予約時間
- reserve_datetime: 予約された時間
- reserve_visitors: 各予約での来客数

### hpg_store_info.csv
- hpg_store_id: HotPepperGourmetの各レストランのid
- hpg_genre_name: HotPepperGourmetでの各レストランのジャンル
- hpg_area_name: HotPepperGourmetの各レストランの都道府県・市町村の情報
- latitude, longitude: 緯度・経度

### store_id_relation.csv
HotPepperGourmetとAirREGIのidの対応テーブル
- hpg_store_id
- air_store_id
