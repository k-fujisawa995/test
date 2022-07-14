import AIModel
import numpy as np
from PIL import Image
 
target_image = "/kasago.jpg"    # テスト用画像
 
im_rows = 150   # 画像の縦ピクセルサイズ
im_cols = 150   # 画像の横ピクセルサイズ
im_color = 3    # 画像の色空間
in_shape = (im_rows, im_cols, im_color)
num_classes = 10 # 分類数
 
LABELS = ["アカムツ", "アマダイ", "オニアジ","カサゴ", "キンメダイ", "ゴマサバ", "タチウオ", "マサバ", "マダイ", "ムラソイ"]   # 分類ラベル数
 
# CNNモデルを取得
model = AIModel.get_model(in_shape, num_classes)
# 保存したモデルを読み込む
 
# 画像を読み込む
img = Image.open(target_image)
img = img.convert("RGB")   # 色空間をRGBに
img = img.resize((im_cols, im_rows))   # サイズ変更
 
# データに変換
x = np.asarray(img)
x = x.reshape(-1, im_rows, im_cols, im_color)
x = x / 255
 
# 予測
pre = model.predict([x])[0]
idx = pre.argmax()
per = int(pre[idx] * 100)
 
print("この写真が", LABELS[idx], "である可能性は", per, "%")