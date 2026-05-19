面向图像检索系统的特征向量安全防护模块

项目简介
本项目实现了一套基于ResNet50特征提取、AES-256-GCM加密存储、SHA-256完整性校验与环境变量密钥管理的安全图像检索方案。
系统支持本地图像库批量特征提取、加密索引加载、安全以图搜图与登录权限控制，在保障检索效率的同时，有效抵御特征向量泄露、篡改与伪造攻击。
项目结构清晰、可复现性强，可用于学术实验复现、安全图像检索系统开发及格式兼容加密检索方向的二次开发。

核心功能
- 基于ResNet50的图像特征批量提取
- AES-256-GCM加密存储特征索引
- SHA-256特征文件完整性校验
- 环境变量安全密钥管理
- 登录权限控制
- Top-K 相似图像检索
- 支持独立特征提取脚本，可离线生成向量文件


项目声明
·项目名称:面向图像检索系统的特征向量安全防护模块设计与实现
·项目作者:Wang Furong
·作者单位:暨南大学网络空间安全学院
·开发语言:Python
·深度学习框架：PyTorch、Torchvision
·前端交互：Gradio
·加密算法：AES-256-GCM、SHA-256
·依赖库：Pillow、NumPy、PyCryptodome、tqdm



使用说明
环境准备
```bash
pip install torch torchvision gradio pillow pycryptodome numpy tqdm
