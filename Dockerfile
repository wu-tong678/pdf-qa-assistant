FROM python:3.10-slim
#用 Python 3.10 精简版作为基础环境
WORKDIR /app
#在容器里建一个 /app 目录，作为工作目录
COPY requirements.txt .
#把依赖文件复制进容器
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
#在容器里装依赖
COPY . .
#把项目所有代码复制进容器
EXPOSE 8000
#声明容器用 8000 端口
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
#