def get_link(id_from, id_to):
    return f"""  <link>
    <from type="int">{id_from}</from>
    <to type="int">{id_to}</to>
    <property name="rel" type="string">peer</property>
    <property name="latitude" type="float">48.20849</property>
    <property name="longitude" type="float">16.37208</property>
    <property name="capacity" type="int">10</property>
    <property name="from_index" type="int">0</property>
    <property name="to_index" type="int">0</property>
  </link>"""


for i in range(1, 50):
    print(get_link(i, i +1))
    
clique = [1, 2, 3, 4, 5]
K = 5

for k in range(K):
    # Add links between all nodes in the clique
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            print(get_link(clique[i], clique[j]))