(Используется python 3.7.9)

1.Переходим в папку в командной строке

2.Создаем виртуальное окружение
python -m venv pc_env

3. Активируем окружение
cd pc_env/Scripts
activate.ps1 			если из powershell (или activate.bat если из cmd)

4. Возвращаемся в исходную папку
cd ../..

5. Устанавливаем библиотеки (нужен git)
pip install -r requirements.txt

или
pip install numpy
python -m pip install git+https://github.com/DanielPollithy/pypcd.git
pip install laspy

5. В shift_rotate.py изменить input_path, output_path, shift_x, shift_y, shift_z, rotation_angle и сохранить

6. Запустить скрипт
python shift_rotate.py
