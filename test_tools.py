'''with open('f1.txt', 'r') as file:
    rows = file.readlines()

k = 0
for i, row in enumerate(rows):
    print(f'{i + 1}', end=' ')
    l = list(map(int, row.split()))
    print(f'l: {l}', end=' ')
    n = {n for n in l if l.count(n) > 1}
    print(f'n: {n}', end=' ')
    s = {n for n in l if l.count(n) == 1}  # уникальные числа
    print(f's: {s}', end=' ')

    if all(map(lambda x: x % 2 == 1, s)) and len(n) + len(s) == 5 and all(map(lambda x: x % 2 == 0, n)):
        k += 1
        print(True, end=' ')
    print()

print(f'k: {k}')'''

'''historys = [['Шанхай', 'Сянган', 'Сеул', 'Мумбаи'],
            ['Шанхай', 'Мумбаи', 'Хошимин'],
            ['Хошимин', 'Мумбаи', 'Джакарта', 'Сянган', 'Шанхай'],
            ['Mocrdf', 'Denver']]

historys = sorted(historys, key=len)

ports = [port for port in historys[0]
         if all(port in history for history in historys[1:])]
ports.sort()

if ports:
    print(*ports)
else:
    print('NO')'''

'''
В текстовом файле, имя которого вводится с клавиатуры, 
находятся строчки, представляющие из себя цепочки из символов латинского алфавита X, I, V, C, M. 
Найдите длину самой длинной подцепочки, не содержащей римское число 17 (XVII), но содержащей римское число сто (C).  
В новый файл output.txt выведите на первой строке найденную максимальную длину, 
а со второй строчки выведите все найденные цепочки максимальной длины - каждая на новой строке. 
'''

'''with open('file1.txt', 'r') as file:
    rows = file.readlines()

result = list()
for row in rows:
    row = row.replace("\n", "")
    print(row)
    while row.find('XVII') != -1:
        i = row.find('XVII')  # конечный индекс вхождения
        print(f'----{row[:i + 3]}')
        if 'C' in row[:i + 3]:
            result.append(row[:i + 3])
        row = row[i + 1:]
    print(f'----{row}')
    if 'C' in row:
        result.append(row)
    print()

max_length = max(len(s) for s in result)
result = [s for s in result if len(s) == max_length]

print(max([len(result) for result in result]))
print(*result, sep='\n')'''

'''
При проведении эксперимента заряженные частицы попадают на чувствительный экран, 
представляющий из себя матрицу размером M на M точек. 
При попадании каждой частицы на экран в протоколе фиксируются координаты попадания: 
номер ряда (целое число от 1 до M) и номер позиции в ряду (целое число от 1 до M). 
Точка экрана, в которую попала хотя бы одна частица, считается светлой, 
точка, в которую ни одна частица не попала, – тёмной. 
При анализе результатов эксперимента рассматривают линии. 
Линией называют группу светлых точек, расположенных в одном ряду подряд, то есть без тёмных точек между ними. 
Линия должна содержать не менее 3 светлых точек, слева и справа от линии должна быть тёмная точка или край экрана. 
Вам необходимо по заданному протоколу определить наибольшее количество линий, расположенных в одном ряду, 
и номер ряда, в котором это количество встречается. 
Если таких рядов несколько, укажите максимально возможный номер.

Входные данные
На вход подается имя файла.
Первая строка входного файла содержит целое число N – общее количество частиц, попавших на экран. 
Вторая строка - размер поля экрана 0<M<=100 000. 
Каждая из следующих N строк содержит 2 целых числа: номер ряда и номер позиции в ряду.

В ответе запишите два целых числа: 
сначала максимальное количество линий в одном ряду, 
затем через пробел – номер ряда, в котором это количество встречается

Ответ: 2 10

Обратите внимание, что в 5 ряду только одна "линия" (остальные участки слишком короткие).

Разные частицы могут прилетать в одну и ту же точку (например, (2;5))
'''

'''
def show_display(d):
    print()
    for row in range(len(d)):
        print(display[row])
    print()


filename = 'f1.txt'
with open(filename, 'r') as file:
    N = int(file.readline())
    M = int(file.readline())
    dots = [list(map(int, line.split())) for line in file]

display = [[' '] * M for row in range(M)]
for dot in dots: display[dot[0] - 1][dot[1] - 1] = '1'

show_display(display)

max_count, max_row = 0, 0
for i, row in enumerate(display):
    s = ''.join(row).split(' ')
    s = [el for el in s if len(el) >= 3]
    if len(s) > max_count:
        max_count, max_row = len(s), i + 1

print(max_count, max_row)
'''
