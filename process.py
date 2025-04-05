import re
import pandas as pd
import matplotlib.pyplot as plt

# 文件路径
input_file_path = "log_5_10_2_4_long.log"
output_file_path = "result_2_11.xlsx"

# 读取文件内容，忽略解码错误
with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as file:
    text = file.read()

# 正则表达式模式
epoch_pattern = r"training epoch (\d+)"
reward_pattern = r"got reward from workers ([\d.]+) seconds"
advantage_pattern = r"advantage ready ([\d.]+) seconds"
gradient_pattern = r"worker send back gradients ([\d.]+) seconds"
average_time_pattern = r"平均时间： ([\d.]+)"
best_time_pattern = r"目前最佳时间： ([\d.]+)"
apply_gradient_pattern = r"apply gradient ([\d.]+) seconds"

# 提取数据
epochs = re.findall(epoch_pattern, text)
rewards = re.findall(reward_pattern, text)
advantages = re.findall(advantage_pattern, text)
gradients = re.findall(gradient_pattern, text)
average_times = re.findall(average_time_pattern, text)
best_times = re.findall(best_time_pattern, text)
apply_gradients = re.findall(apply_gradient_pattern, text)

# 获取最小数据长度以确保数据完整性
min_length = min(len(epochs), len(rewards), len(advantages), len(gradients), len(average_times), len(best_times), len(apply_gradients))

# 创建数据字典并转换数值类型
data = {
    "Epoch": [int(epoch) for epoch in epochs[:min_length]],
    "Reward Time (s)": [float(reward) for reward in rewards[:min_length]],
    "Advantage Ready Time (s)": [float(adv) for adv in advantages[:min_length]],
    "Gradient Send Back Time (s)": [float(grad) for grad in gradients[:min_length]],
    "Average Time (s)": [float(avg) for avg in average_times[:min_length]],
    "Best Time (s)": [float(best) for best in best_times[:min_length]],
    "Apply Gradient Time (s)": [float(apply) for apply in apply_gradients[:min_length]],
}

# 创建DataFrame并保存为Excel
df = pd.DataFrame(data)
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='Data')

    # 获取工作簿和工作表对象
    workbook = writer.book
    worksheet = writer.sheets['Data']

    # 创建图表对象
    chart = workbook.add_chart({'type': 'line'})

    # 配置图表数据源
    max_row = len(df) + 1
    chart.add_series({
        'name': 'Average Time (s)',
        'categories': f'=Data!$A$2:$A${max_row}',
        'values': f'=Data!$E$2:$E${max_row}',
        'line': {'color': 'blue'}
    })
    chart.add_series({
        'name': 'Best Time (s)',
        'categories': f'=Data!$A$2:$A${max_row}',
        'values': f'=Data!$F$2:$F${max_row}',
        'line': {'color': 'red'}
    })

    # 添加标题和轴标签
    chart.set_title({'name': 'Average and Best Time per Epoch'})
    chart.set_x_axis({'name': 'Epoch'})
    chart.set_y_axis({'name': 'Time (s)'})

    # 插入图表到工作表
    worksheet.insert_chart('H2', chart)

print(f"数据和图表已成功保存到 {output_file_path}")

