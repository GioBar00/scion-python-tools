count = 0
for i in range(30042,30052):
    print(f'- name: dispatcher{count}\n    port: {i}\n    protocol: UDP\n    targetPort: {i}')
    count+=1