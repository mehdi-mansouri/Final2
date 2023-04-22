from Doc2vecClass1 import Doc2vecClass
from entityClassificaton import entityClass
doc=Doc2vecClass(['haus','space','buy','money','payment'],False)
liste=doc.userText()
for item in liste:
    print(item)
    entity=entityClass(item)
    entity.entityspacy()
    entity.entityNtlk()
    print('---...----')
#TODO hier sollte man nur intentclassification auch hinzufügen ,aber
# gibt es problem bei INPUT und INTENT 