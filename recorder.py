import sqlite3

class Recorder:
    def __init__(self):
        self.con = sqlite3.connect("results.db")
        cur = self.con.cursor()
        with open("results_scheme.sql", 'r') as file:
            scheme = file.read()
        cur.executescript(scheme)

    def dataset(self, label, description):
        cur = self.con.cursor()
        cur.execute('INSERT OR IGNORE INTO dataset (label, description) VALUES (?,?)', (label, description))
        cur.execute('SELECT id FROM dataset WHERE label = ? AND description = ?', (label, description))
        return cur.fetchone()[0]

    def model(self, label, description, trainable_params):
        cur = self.con.cursor()
        cur.execute('INSERT OR IGNORE INTO model (label, description, trainable_params) VALUES (?,?,?)', (label, description, int(trainable_params)))
        cur.execute('SELECT id FROM model WHERE label = ? AND description = ?', (label, description))
        return cur.fetchone()[0]

    def optimizer(self, label, lr, all_parameters):
        cur = self.con.cursor()
        cur.execute('INSERT OR IGNORE INTO optimizer (label, lr, all_parameters) VALUES (?,?,?)', (label, str(lr), str(all_parameters)))
        cur.execute('SELECT id FROM optimizer WHERE label = ? AND lr = ? AND all_parameters = ?', (label, str(lr), str(all_parameters)))
        return cur.fetchone()[0]

    def criterion(self, label):
        cur = self.con.cursor()
        cur.execute('INSERT OR IGNORE INTO criterion (label) VALUES (?)', (label,))
        cur.execute('SELECT id FROM criterion WHERE label = ?', (label,))
        return cur.fetchone()[0]

    def run(self, dataset, model, optimizer, criterion, seed, epochs, batch_size):
        ds_id = self.dataset(**dataset)
        m_id = self.model(**model)
        o_id = self.optimizer(**optimizer)
        c_id = self.criterion(**criterion)
        cur = self.con.cursor()
        cur.execute('INSERT INTO run (dataset, model, optimizer, criterion, seed, epochs, batch_size) VALUES (?,?,?,?,?,?,?)', (ds_id, m_id, o_id, c_id, seed, epochs, batch_size))
        return cur.lastrowid

    def max_ram(self, run_id, max_ram):
        self.con.cursor().execute('UPDATE run SET max_ram = ? WHERE id = ?', (str(max_ram), run_id))

    def min_loss(self, run_id, min_loss):
        self.con.cursor().execute('UPDATE run SET min_loss = ? WHERE id = ?', (str(min_loss), run_id))

    def step(self, run_id, epoch, iteration, loss, wall_clock_time):
        self.con.cursor().execute('INSERT INTO step (run, epoch, iteration, loss, wall_clock_time) VALUES (?, ?, ?, ?, ?)', (str(run_id), str(epoch), str(iteration), str(loss), str(wall_clock_time)))

    def commit(self):
        self.con.commit()

    def __del__(self):
        self.con.commit()
        self.con.close()
