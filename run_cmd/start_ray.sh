
ray start --head --port=65379 --dashboard-host=0.0.0.0 --dashboard-port=8872 # 当前节点为头节点 启动Ray

ray start --head --port=65379 --dashboard-host=0.0.0.0 --dashboard-port=8872 --num-cpus=64 --num-gpus=8 # 限制GPU, CPU 资源


# pkill -9 -f "ray::" && rm -rf /tmp/ray && ray stop --force # 停止Ray

# ray start --address='10.18.33.9:65379' --dashboard-host=0.0.0.0 --dashboard-port=8872 # 新节点加入到已有头节点
