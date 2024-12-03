BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "criterion" (
	"id"	INTEGER,
	"label"	TEXT NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT),
  UNIQUE("label")
);
CREATE TABLE IF NOT EXISTS "dataset" (
	"id"	INTEGER,
	"label"	TEXT NOT NULL,
	"description"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT),
  UNIQUE("label", "description")
);
CREATE TABLE IF NOT EXISTS "model" (
	"id"	INTEGER,
	"label"	TEXT NOT NULL,
	"trainable_params" TEXT NOT NULL,
	"description"	TEXT NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT),
  UNIQUE("label", "description")
);
CREATE TABLE IF NOT EXISTS "optimizer" (
	"id"	INTEGER,
	"label"	TEXT NOT NULL,
	"lr"	NUMERIC NOT NULL,
	"all_parameters"	TEXT NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT),
  UNIQUE("label", "lr", "all_parameters")
);
CREATE TABLE IF NOT EXISTS "run" (
	"id"	INTEGER,
	"dataset"	INTEGER NOT NULL,
	"model"	INTEGER NOT NULL,
	"optimizer"	INTEGER NOT NULL,
	"seed"	INTEGER NOT NULL,
	"criterion"	INTEGER NOT NULL,
	"epochs"	INTEGER NOT NULL,
	"batch_size"	INTEGER NOT NULL,
	"min_loss"	NUMERIC,
	"max_ram"	NUMERIC,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("criterion") REFERENCES "criterion"("id"),
	FOREIGN KEY("dataset") REFERENCES "dataset"("id"),
	FOREIGN KEY("model") REFERENCES "model"("id"),
	FOREIGN KEY("optimizer") REFERENCES "optimizer"("id")
);
CREATE TABLE IF NOT EXISTS "step" (
	"id"	INTEGER,
	"run"	INTEGER NOT NULL,
	"epoch"	INTEGER NOT NULL,
	"iteration"	INTEGER NOT NULL,
	"loss"	NUMERIC NOT NULL,
	"wall_clock_time"	INTEGER NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("run") REFERENCES "run"("id") ON DELETE CASCADE,
  UNIQUE("run", "epoch", "iteration")
);
COMMIT;
