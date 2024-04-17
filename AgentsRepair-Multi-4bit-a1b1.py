# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import subprocess
import os
import gc
from tqdm import tqdm
from torch.nn.parallel import DataParallel

model1_id = "codellama/CodeLlama-7b-Instruct-hf"
model2_id = "mistralai/Mistral-7B-Instruct-v0.2"
file_path = "./SPoC/test-testp.txt"
testcases_base_path = './SPoC/testcases'

device = "cuda"

tokenizer1 = AutoTokenizer.from_pretrained(model1_id)
model1 = AutoModelForCausalLM.from_pretrained(
    model1_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)

tokenizer2 = AutoTokenizer.from_pretrained(model2_id)
model2 = AutoModelForCausalLM.from_pretrained(
    model2_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)

with open(file_path, "r", encoding="utf-8") as file:
    pseudocode_content = file.read()

# 将伪代码拆分为每个probid的部分
pseudocodes = pseudocode_content.split("\n\n")

# 设置从第几个伪代码开始生成
success_count = 0
start_index = 0
passed_tests = 0
total_tests = 0
float1 = 0
float2 = 0
passed_rate = 0

# 统计伪代码的数量
num_pseudocodes = len(pseudocodes)
if not pseudocodes[-1].strip():
    num_pseudocodes -= 1

print(f"文件中一共有 {num_pseudocodes} 个伪代码。")

# 使用 tqdm 创建进度条，设置 initial 参数
progress_bar = tqdm(total=num_pseudocodes, desc="Generating Programs", initial=start_index)


def extract_cpp_code(terminal_output):
    # 匹配 probid 行
    pattern_probid = r'probid:\s*(\w+)'
    match_probid = re.search(pattern_probid, terminal_output)

    if match_probid:
        probid_content = match_probid.group(1).strip()
    else:
        print("未找到匹配的 probid: 行.")
        return None, None, None

    # 匹配所有代码块
    pattern_code = r'```(.*?)```'
    matches_code = re.findall(pattern_code, terminal_output, re.DOTALL)

    cpp_code = ""

    # 从最后一个代码块开始遍历
    for code_block in reversed(matches_code):
        if '#include <iostream>' in code_block:
            lines = code_block.split('\n')
            if all('```' not in line for line in lines[:-1]):  # 排除最后一行含有```的情况
                code_lines = code_block.split('\n')
                if code_lines[0].strip().startswith(('cpp', 'c++')):
                    code_block = '\n'.join(code_lines[1:])
                cpp_code = code_block.strip()
                break

    if cpp_code:
        print(f"cppcode:\n{cpp_code}")
        return True, cpp_code, probid_content
    else:
        print("未找到符合条件的 C++ 代码块。")
        return False, None, probid_content


def compile_and_run_cpp(cpp_code):
    with open('temp.cpp', 'w') as file:
        file.write(cpp_code)
    # 仅编译一次
    compile_process = subprocess.run(['g++', 'temp.cpp', '-o', 'temp'], capture_output=True, text=True)
    if compile_process.returncode != 0:
        return False, compile_process.stderr
    return True, None


def read_test_cases(probid):
    test_cases = []
    testcases_base_path = '/home/yewenguang/work/Code-Llama/spoc/testcases'
    testcases_path = f"{testcases_base_path}/{probid}/{probid}_testcases_public.txt"
    print("正在进行测试用例测试：")
    print(f"testcases_path: {testcases_path}")
    with open(testcases_path, 'r') as file:
        lines = file.readlines()
        input_part = []
        output_part = []
        reading_input = True

        for line in lines:
            if line.strip() == "###ENDINPUT###":
                reading_input = False
            elif line.strip() == "###ENDOUTPUT###":
                test_cases.append((input_part, output_part))
                input_part = []
                output_part = []
                reading_input = True
            elif reading_input:
                input_part.append(line.strip())
            else:
                output_part.append(line.strip())
    return test_cases


def run_test_cases(cpp_code, probid):
    global passed_tests, total_tests
    with open('temp.cpp', 'w') as file:
        file.write(cpp_code)
    test_cases = read_test_cases(probid)
    flag = 0
    failed_test_cases = []
    for idx, (input_data, expected_output) in enumerate(test_cases):
        output = run_cpp_with_input(cpp_code, input_data)
        print(f"output: {output}")
        print(f"expected_output: {expected_output}")
        if compare_output(output, expected_output):
            print(f"Test case {idx + 1}: PASSED")
            flag += 1
        else:
            print(f"Test case {idx + 1}: FAILED")
            input_data_str = ''.join(input_data)
            expected_output_str = ''.join(expected_output)
            failed_test_cases.append((input_data_str, output, expected_output_str))
            flag = 0
            break
    if flag != 0:
        return True, None
    else:
        return False, failed_test_cases[0]


def run_cpp_with_input(code, input_data):
    try:
        process = subprocess.Popen('./temp', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        stdout, stderr = process.communicate(input='\n'.join(input_data), timeout=10)  # 将输入数据转换为字符串形式并传递给子进程
        print(f"input_data: {input_data}")
        # 检查输出中是否存在冒号
        last_colon_index = stdout.rfind(":")  # 找到最后一个冒号的位置
        if last_colon_index != -1:
            return stdout[last_colon_index + 1:].strip()  # 返回最后一个冒号后的部分
        else:
            return stdout.strip()  # 返回完整输出

    except Exception as e:
        return f"运行时出错: {e}"


def compare_output(output, expected_output):
    return output == '\n'.join(expected_output)


def extract_error_info(error_messages, source_code):
    error_info = ""

    # 使用正则表达式匹配错误信息中的行号、错误内容以及错误指示符位置
    pattern = re.compile(r'<stdin>:(\d+):(\d+): (error|note): (.+)')
    matches = pattern.findall(error_messages)

    for match in matches:
        line_num = int(match[0])
        error_desc = match[2]

        if error_desc == 'error':
            error_content = source_code.split('\n')[line_num - 1].strip()
            error_info += f'error: Line: {line_num}, Line_content: "{error_content}", error_msg: "{match[3]}"\n'
        else:
            error_info += f'note: Line: {line_num}, note_msg: "{match[3]}"\n'

    return error_info


failure_indices = []
probid_content = ''
success_rate = 0
first_rate = 0
second_rate = 0

for idx, pseudocode in enumerate(pseudocodes[start_index:-1]):
    success = False
    cpp_code = ""
    epochs = 0

    test_pass = False
    compile_pass = False

    while epochs < 1 and not test_pass:
        # 代码生成
        consecutive_failures = 0  # 记录连续失败次数
        while not success:
            user_query = f"""
Convert the following pseudocode to C++code:
{pseudocode}
"""

            prompt = f"<s>[INST] {user_query.strip()} [/INST]\n"
            inputs = tokenizer1(prompt, return_tensors="pt").to("cuda")

            output = model1.generate(
                inputs["input_ids"],
                max_new_tokens=4096,
                do_sample=True,
                top_p=0.9,
                temperature=0.1,
            )
            output = output[0].to("cpu")
            print(tokenizer1.decode(output))

            success, cpp_code, probid_content = extract_cpp_code(tokenizer1.decode(output))

        print("Codellama: Code generation finished")

        # while not compile_pass or not test_pass:
        while not compile_pass or not test_pass:
            compile_pass, error_message = compile_and_run_cpp(cpp_code)
            if compile_pass:
                test_pass, failed_test_cases = run_test_cases(cpp_code, probid_content)
                if test_pass:
                    float1 += 1
                    break  # 如果成功提取 C++ 代码块，退出循环
                else:
                    limited_lines = []
                    limited_text = ""
                    if failed_test_cases is not None:
                        for line in failed_test_cases[1].strip().split('\n')[:3]:
                            limited_lines.append(line[:10])

                        limited_text = '\n'.join(limited_lines)
                    success = False
                    while not success:
                        user_query = f"""
### Instruction:
The following is pseudocode and incorrect C++program translation. 
Compare pseudo program and C++ program line by line, analyze the differences in functionality between the two, and provide a summary in sections.
Finally, make corrections based on the summary and feedback provided.

### Information Provided:
{pseudocode}

C++ program:
{cpp_code}

Feedback:
Wrong Answer with input: "{failed_test_cases[0]}". 
Expected output is "{failed_test_cases[2]}", but generated output is "{limited_text}". 

### Request:
You should provide the complete modified C++code in the end.
"""
                        prompt = f"<s>[INST] {user_query.strip()} [/INST]\n"
                        inputs = tokenizer1(prompt, return_tensors="pt").to("cuda")

                        output = model1.generate(
                            inputs["input_ids"],
                            max_new_tokens=4096,
                            do_sample=True,
                            top_p=0.9,
                            temperature=0.1,
                        )
                        output = output[0].to("cpu")
                        print(tokenizer1.decode(output))
                        success, cpp_code2, probid_content = extract_cpp_code(tokenizer1.decode(output))

                        if success:
                            cpp_code = cpp_code2

                    print(
                        "Codellama: The test case did not pass, and we analyzed the reasons and modification methods.")

                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        print("codellama连续三次失败，将跳过codellama")
                        break  # 如果连续三次失败，跳出循环，执行下一次循环
            else:
                Error_Message = ""
                if error_message is not None:
                    Error_Message = extract_error_info('\n'.join(error_message.strip().split('\n')[:50]), cpp_code)

                success = False
                while not success:
                    user_query = f"""
### Instruction:
There was some errors during the compilation of the C++program. 
First, you should review the pseudocode and error messages corresponding to the C++program.  Then propose the best solution for each error, fix the bugs. And then you should provide the complete modified C++code in the end.
Enclose the provided C++ code snippet within triple backticks ``` ``` to properly format the code block.

### Information Provided: 
C++ program:
{cpp_code}

{pseudocode}

error_message:
{Error_Message}
Fix the bug.

### Request:
You should provide the complete modified C++code in the end.
"""
                    prompt = f"<s>[INST] {user_query.strip()} [/INST]\n"
                    inputs = tokenizer1(prompt, return_tensors="pt").to("cuda")

                    output = model1.generate(
                        inputs["input_ids"],
                        max_new_tokens=4096,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.1,
                    )
                    output = output[0].to("cpu")
                    print(tokenizer1.decode(output))

                    success, cpp_code1, probid_content = extract_cpp_code(tokenizer1.decode(output))
                    if success:
                        cpp_code = cpp_code1
                print("Codellama: The compile did not pass, and we analyzed the reasons and modification methods.")

                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print("codellama连续三次失败，将跳过codellama")
                    break  # 如果连续三次失败，跳出循环，执行下一次循环

        epochs += 1

    if not compile_pass and not test_pass:
        compile_pass, error_message = compile_and_run_cpp(cpp_code)
        if compile_pass:
            test_pass, failed_test_cases = run_test_cases(cpp_code, probid_content)
            if test_pass:
                float1 += 1


    while not test_pass and 1 <= epochs < 2:
        consecutive_failures = 0
        success = False
        while not success:
            messages = [
                {
                    "role": "user",
                    "content": f"""
Convert the following pseudocode to C++code:
{pseudocode}
"""
                },
            ]
            encodeds = tokenizer2.apply_chat_template(messages, return_tensors="pt")

            model_inputs = encodeds.to(device)


            generated_ids = model2.generate(model_inputs, max_new_tokens=4096, do_sample=True)
            decoded = tokenizer2.batch_decode(generated_ids)
            # print(decoded[0])

            success, cpp_code, probid_content = extract_cpp_code(decoded[0])
        print("Mistral: Code generation finished")

        test_pass = False
        compile_pass = False

        # while not compile_pass or not test_pass:
        while not compile_pass or not test_pass:
            compile_pass, error_message = compile_and_run_cpp(cpp_code)
            if compile_pass:
                test_pass, failed_test_cases = run_test_cases(cpp_code, probid_content)
                if test_pass:
                    float2 += 1
                    break  # 如果成功提取 C++ 代码块，退出循环
                else:
                    limited_lines = []
                    limited_text = ""
                    if failed_test_cases is not None:
                        for line in failed_test_cases[1].strip().split('\n')[:3]:
                            limited_lines.append(line[:10])

                        limited_text = '\n'.join(limited_lines)
                    success = False
                    while not success:
                        messages = [
                            {
                                "role": "user",
                                "content": f"""
### Instruction:
The following is pseudocode and incorrect C++program translation. 
Compare pseudo program and C++ program line by line, analyze the differences in functionality between the two, and provide a summary in sections.
Finally, make corrections based on the summary and feedback provided.

### Information Provided:
{pseudocode}

C++ program:
{cpp_code}

Feedback:
Wrong Answer with input: "{failed_test_cases[0]}". 
Expected output is "{failed_test_cases[2]}", but generated output is "{limited_text}". 

### Request:
You should provide the complete modified C++code in the end.
"""
                            },
                        ]

                        encodeds = tokenizer2.apply_chat_template(messages, return_tensors="pt")

                        model_inputs = encodeds.to(device)


                        generated_ids = model2.generate(model_inputs, max_new_tokens=4096, do_sample=True)
                        decoded = tokenizer2.batch_decode(generated_ids)
                        print(decoded[0])

                        success, cpp_code2, probid_content = extract_cpp_code(decoded[0])
                        if success:
                            cpp_code = cpp_code2
                    print("Mistral: The test case did not pass, and we analyzed the reasons and modification methods.")

                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        print("mistral连续三次提取失败，将跳过此伪代码。")
                        break  # 如果连续三次失败，跳出循环，执行下一次循环
            else:
                Error_Message = ""
                if error_message is not None:
                    Error_Message = extract_error_info('\n'.join(error_message.strip().split('\n')[:50]), cpp_code)

                success = False
                while not success:
                    messages = [
                        {
                            "role": "user",
                            "content": f"""
### Instruction:
There was some errors during the compilation of the C++program. 
First, you should review the pseudocode and error messages corresponding to the C++program.  Then propose the best solution for each error, fix the bugs. And then you should provide the complete modified C++code in the end.
Enclose the provided C++ code snippet within triple backticks ``` ``` to properly format the code block.

### Information Provided: 
C++ program:
{cpp_code}

{pseudocode}

error_message:
{Error_Message}
Fix the bug.

### Request:
You should provide the complete modified C++code in the end.
"""
                        },
                    ]

                    encodeds = tokenizer2.apply_chat_template(messages, return_tensors="pt")

                    model_inputs = encodeds.to(device)


                    generated_ids = model2.generate(model_inputs, max_new_tokens=4096, do_sample=True)
                    decoded = tokenizer2.batch_decode(generated_ids)
                    print(decoded[0])

                    success, cpp_code1, probid_content = extract_cpp_code(decoded[0])

                    if success:
                        cpp_code = cpp_code1
                print("Mistral: The compile did not pass, and we analyzed the reasons and modification methods.")

                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print("mistral连续三次提取失败，将跳过此伪代码。")
                    break  # 如果连续三次失败，跳出循环，执行下一次循环
        epochs += 1

    if not compile_pass and not test_pass:
        compile_pass, error_message = compile_and_run_cpp(cpp_code)
        if compile_pass:
            test_pass, failed_test_cases = run_test_cases(cpp_code, probid_content)
            if test_pass:
                float2 += 1

    if success:
        success_count += 1
    else:
        failure_indices.append(idx)

    if test_pass:
        passed_tests += 1

    total_tests += 1

    # print(f"success_count:{success_count}, passed_tests:{passed_tests}, total_tests:{total_tests}")
    passed_rate = (passed_tests / total_tests) * 100
    success_rate = (success_count / total_tests) * 100
    first_rate = (float1 / total_tests) * 100
    second_rate = (float2 / total_tests) * 100
    progress_bar.update(1)
    progress_bar.set_postfix({
        "success_count": success_count,
        "passed_tests:": passed_tests,
        "total_tests": total_tests,
        "主模型通过数": float1,
        "副模型通过数": float2,
        "生成成功率": f"{success_rate:.2f}%",
        "测试成功率": f"{passed_rate:.2f}%",
        "主模型通过率": f"{first_rate:.2f}%",
        "副模型通过率": f"{second_rate:.2f}%"
    })
    gc.collect()
    torch.cuda.empty_cache()

progress_bar.set_postfix({
    "success_count": success_count,
    "passed_tests:": passed_tests,
    "total_tests": total_tests,
    "主模型通过数": float1,
    "副模型通过数": float2,
    "生成成功率": f"{success_rate:.2f}%",
    "测试成功率": f"{passed_rate:.2f}%",
    "主模型通过率": f"{first_rate:.2f}%",
    "副模型通过率": f"{second_rate:.2f}%"
})
# 关闭进度条
progress_bar.close()

print(f"测试成功率:{passed_rate:.2f}%")
print(f"成功提取的 C++ 代码块数量：{success_count}/{num_pseudocodes}")
print(f"提取失败的索引：{failure_indices}")