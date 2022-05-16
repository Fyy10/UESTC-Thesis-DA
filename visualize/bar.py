import matplotlib.pyplot as plt

plt.figure('Ablation_Study')
plt.title('Effect of the Key Components')
exp = [1, 2, 3, 4]
acc = [44.6, 46.0, 51.3, 52.6]
plt.bar(exp, acc, width=0.5, color=['r', 'g', 'b', 'y'])
plt.ylabel('Accuracy')
plt.ylim((40, 60))
plt.xlabel('Model Components')
plt.xticks(exp, labels=['$L_s$', '$L_s+$FSM', '$L_s+L_{km}$', r'$L_s + $FSM $+\ L_{km}$'])
for i in exp:
    plt.text(i, acc[i-1], str(acc[i-1]), ha='center', va='bottom', fontsize=10)
plt.show()
