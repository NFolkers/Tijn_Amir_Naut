# what we build

### a react frontend for our api

runnable with the command: npm run dev -- --port 3000
run npm install before running this to get the packages required
this has to be executed in the /app/frontend/IML_frontend directory

### a fastapi backend where you can upload an image and our model responds with the classification

- runnable with the command: uvicorn main:app --reload --host=localhost
- this has to be executed in the main /app directory

- both require a seperate running instance in a seperate terminal
- on the outputted ip from the frontend it should display a button to upload a file and send it for classification

- out docker compose should be done but it somehow does not compile, this might be resolvable still
