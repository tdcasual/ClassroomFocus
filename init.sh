#!/usr/bin/env bash
# Ubuntu 24.04 专用：课堂专注度分析仪项目初始化脚本
# 包含：系统依赖安装 + Python venv + requirements + 目录结构

set -euo pipefail


##############################################
# 0) 检查 Ubuntu 版本
##############################################
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "🔎 系统：$PRETTY_NAME"
    if [[ "$VERSION_ID" != "24.04" ]]; then
        echo "⚠️ 警告：检测到不是 Ubuntu 24.04，本脚本为 24.04 专用。"
    fi
fi


##############################################
# 1) 安装系统依赖（OpenCV、MediaPipe、音频等）
##############################################
echo "📦 Step 1：安装系统依赖..."

sudo apt update

sudo apt install -y \
    python3 python3-venv python3-pip \
    build-essential cmake pkg-config \
    libgl1 libglib2.0-0 \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libxvidcore-dev libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev gfortran \
    portaudio19-dev pulseaudio-utils \
    libssl-dev libffi-dev

echo "✅ 系统依赖安装完成"
echo ""


##############################################
# 2) 创建项目目录
##############################################
PROJECT_NAME="classroom_focus"

echo "📁 Step 2：创建项目结构..."

mkdir -p "$PROJECT_NAME"/{cv,asr,sync,replay,viz}
touch "$PROJECT_NAME"/{main_pc.py,main_pi.py,config.py,requirements.txt}

for dir in cv asr sync replay viz; do
    touch "$PROJECT_NAME/$dir/__init__.py"
done

echo "✅ 目录结构创建完成：$PROJECT_NAME"
echo ""


##############################################
# 3) 创建 Python 虚拟环境（Ubuntu 24.04 自带 Python 3.12）
##############################################
echo "🐍 Step 3：创建虚拟环境..."

PY_BIN="python3"

PY_VER=$($PY_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

echo "ℹ️ 当前 Python 版本：$PY_VER"

if [[ "$PY_VER" != "3.12" ]]; then
    echo "⚠️ 警告：Ubuntu 24.04 默认 Python 是 3.12，你现在的是 $PY_VER"
    echo "   MediaPipe 官方支持 Python 3.9–3.12，请确认你的系统没有改动 Python"
fi

cd "$PROJECT_NAME"
$PY_BIN -m venv .venv
source .venv/bin/activate

echo "✅ 虚拟环境已创建并激活"
echo ""


##############################################
# 4) 生成 requirements.txt
##############################################
echo "📦 Step 4：写入 requirements.txt..."

cat > requirements.txt <<EOF
opencv-python>=4.7,<5.0
mediapipe>=0.10.13
streamlit
websocket-client
numpy
sounddevice
tqdm
python-dotenv
requests
EOF

echo "📥 安装依赖（可能需要几分钟）..."

pip install --upgrade pip

if ! pip install -r requirements.txt; then
    echo "❌ 依赖安装失败，请检查网络或切换 pip 镜像，例如："
    echo "    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/"
    exit 1
fi

echo "✅ 依赖安装成功"
echo ""


##############################################
# 完成！
##############################################
echo "🎉 初始化完成！你现在可以开始开发了。"
echo ""
echo "👉 进入项目："
echo "    cd $PROJECT_NAME"
echo "    source .venv/bin/activate"
echo ""
echo "👉 开发入口："
echo "    python main_pc.py"
echo ""

