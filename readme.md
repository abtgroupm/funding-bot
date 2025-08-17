docker compose build
docker build -t funding-bot .
docker run --env-file .env funding-bot

docker compose stop funding-bot
docker compose down
docker compose up -d # run all doker
docker compose up -d funding-bot # run docker name funding-bot
# show the result
docker compose logs -f fundingbot
docker compose logs -f
# show list container
docker ps -a