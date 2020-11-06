# 反编译出pyinstaller 打包的exe文件
#### 基本步骤
1. 生成exe文件
``` 
    pyinstaller -F main.py
 ```
2. 使用解包程序
[pyinstxtractor.py](https://github.com/countercept/python-exe-unpacker/blob/master/pyinstxtractor.py)
![网页截图](./../../QQ截图20201105181446.png)

3. cmd执行此命令
```
$C:\ python.exe pyinstxtractor.py main.exe
```
4. 将解包得到的pyc字节码文件借助[此网站](https://python-decompiler.com/)反编译
https://python-decompiler.com/
![网页截图](./../../QQ截图20201105181018.png

