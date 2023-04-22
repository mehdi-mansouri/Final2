import asyncio
from Doc2vecClass1 import Doc2vecClass
from TextClass import TextArbeiten
from entityClassificaton import entityClass
from IntentKlassifikation import intent
from nio import AsyncClient, MatrixRoom, RoomMessageText


async def message_callback(room: MatrixRoom, event: RoomMessageText) -> None:
    print(
        f"Message received in room {room.display_name}\n"
        f"{room.user_name(event.sender)} | {event.body}"
    )
    if room.user_name(event.sender) != 'mehdimansouri1':
        print(event.body)
        sentence = event.body
        # word_list = sentence.split()
        user_text = TextArbeiten(event.body)
        text = user_text.text_bearbeiten()
        text_user=text
        text = text.split()
        entity_user=[]
        intent_user=[]
        intent_result_array=[]

        entity_result=[]
        print(text)
        # hier sollte auch Sentiment als 3. Parametr hinzufügen
        doc = Doc2vecClass(text, False,'posetive')

        liste = doc.userText()
        print('-----------entity user text---------')
       
        # -----------Intent user Text
        entity_user_text = entityClass(text_user)
        #print(entity_user_text.entityspacy())
        entity_user.append(entity_user_text.entityspacy())
        
        print('-----------entity user text---------')
        print('-----------Intent user text---------')
        intent_user_text=intent()
        intent_user.append(intent_user_text.intentText(text_user))
        print('Intent of User Text',intent_user)

        client = AsyncClient("https://matrix.org", "mehdimansouri1")
        await client.login("Taghi1993!")
        intent_result=intent()
        for item in liste:
            print('-------------------------------')
            entity = entityClass(item)
            entity.entityspacy()
            entity_result.append(entity.entityspacy())  
            intent_result_array.append(intent_result.intentText(item))
            
            await client.room_send(
                room_id="!TiZQWJRecqKckzPCMQ:matrix.org",
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": item},
            )
        print(entity_result)
        print('------------intent')
        print(intent_result_array)

async def main() -> None:
    client = AsyncClient("https://matrix.org", "mehdimansouri1")
    client.add_event_callback(message_callback, RoomMessageText)
    print(await client.login("Taghi1993!"))

    await client.sync_forever(timeout=300000)  # milliseconds
asyncio.get_event_loop().run_until_complete(main())
