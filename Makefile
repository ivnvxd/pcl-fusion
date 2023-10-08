pcl:
	poetry run python main.py -p

cal:
	poetry run python main.py -c

board:
	poetry run python utils/charuco_board.py

test:
	poetry run pytest --cov --cov-report term-missing

cov:
	poetry run pytest --cov --cov-report xml