import subprocess
import time

def main():
    # 定义你要执行的 PowerShell 命令
    # 注意：在 Python 字符串中，反斜杠 \ 需要转义，所以写成 .\\weight.json
    cmd = 'Get-Content .\\weight.json -Raw | .\\solver.exe'
    
    total_runs = 20
    print(f"开始自动执行 {total_runs} 次...")
    start_time = time.time()

    for i in range(1, total_runs + 1):
        print(f"[{i}/{total_runs}] 正在执行...", end="", flush=True)
        # 使用 subprocess 调用 powershell 执行命令
        # capture_output=True 可以捕获输出，防止控制台被刷屏
        result = subprocess.run(
            ["powershell", "-Command", cmd],
            capture_output=True,
            text=True,
            encoding='gbk',      # Windows 终端默认通常是 GBK 编码，防止中文报错
            errors='ignore'      # 忽略无法解码的字符
        )
        
        if result.returncode == 0:
            print(" 成功！")
            # 如果你想看 solver.exe 输出在屏幕上的内容，可以取消下面这行的注释
            print(result.stdout)
            # R = result.stdout[]  # 获取输出结果并去掉首尾空白字符
            # R_list = result.stdout["R"]
            # 先获取数据后重新微调模型
            # todo: 获取数据后重新微调代理模型

            # todo:更新weight.json获取新数据
            
        else:
            print(" 失败！")
            print(f"错误信息: {result.stderr}")
            
        # 如果你希望每次执行之间稍微停顿一下（例如 0.5 秒），可以取消注释下面这行
        # time.sleep(0.5)

        # 可以删掉usage.count文件重置计数

    end_time = time.time()
    print(f"\n全部 {total_runs} 次执行完毕！总耗时: {end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    main()