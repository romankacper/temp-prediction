CREATE TABLE train (
  date varchar(10),
  meantemp double,
  humidity double,
  wind_speed double,
  meanpressure double
);

CREATE TABLE test (
  date varchar(10),
  meantemp double,
  humidity double,
  wind_speed double,
  meanpressure double
);

LOAD DATA INFILE '/docker-entrypoint-initdb.d/0_Train.csv' 
INTO TABLE train 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n' 
IGNORE 1 ROWS;
SHOW WARNINGS;

LOAD DATA INFILE '/docker-entrypoint-initdb.d/1_Test.csv' 
INTO TABLE test 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n' 
IGNORE 1 ROWS;
SHOW WARNINGS;