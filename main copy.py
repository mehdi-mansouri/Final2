
import asyncio
from nio import  AsyncClient, MatrixRoom, RoomMessageText

async def message_callback(room: MatrixRoom, event: RoomMessageText) -> None:
    print(
        f"Message received in room {room.display_name}\n"
        f"{room.user_name(event.sender)} | {event.body}"
    )
    if event.body =='Hallo':
        client = AsyncClient("https://matrix.org", "mehdimansouri1")
        await client.login("Taghi1993!")
        await client.room_send(
            room_id="!TiZQWJRecqKckzPCMQ:matrix.org",
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": "Hallo,was kann ich für Sie tun?"},
        )

async def main() -> None:
    client = AsyncClient("https://matrix.org", "mehdimansouri1")
    client.add_event_callback(message_callback, RoomMessageText)
    print(await client.login("Taghi1993!"))

    await client.sync_forever(timeout=300000)  # milliseconds
asyncio.get_event_loop().run_until_complete(main())