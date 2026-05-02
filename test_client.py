import asyncio, json
import websockets

async def main():
    async with websockets.connect('ws://localhost:8765', max_size=10*1024*1024) as ws:
        init = json.loads(await ws.recv())
        print('INIT',init.get('type'),init.get('stage'))
        # send a short text message
        await ws.send(json.dumps({'type':'text_msg','text':'Who are you?'}))
        # read a few messages
        for i in range(6):
            m = json.loads(await ws.recv())
            print('MSG',m.get('type'), m.get('text') if 'text' in m else m)

if __name__=='__main__':
    asyncio.run(main())
