# Git 伺服器資訊
hostname=https://biicgitlab.ee.nthu.edu.tw/
port=22

# 取得自己 OS 中放置 CA 的位置
trust_cert_file_location=`curl-config --ca`

# 匯入 Git 伺服器憑證
sudo bash -c "echo -n | openssl s_client -showcerts -connect $hostname:$port \
    2>/dev/null  | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p'  \
    >> $trust_cert_file_location"


