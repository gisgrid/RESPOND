import sqlite3
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta
from respond.scenario.utils import remove_actions_from_string


class DBLevel2RiskPattern:


    def __init__(self, database: str) -> None:
 
        self.database = database

    def createTable(self) -> None:

        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS level2RiskPattern (
                level2_pattern INTEGER NOT NULL,
                risk_pattern TEXT,
                action_id INTEGER,
                dont_action_id_1 INTEGER,
                dont_action_id_2 INTEGER,
                dont_action_id_3 INTEGER,
                ts_first TEXT,
                ts_last TEXT,
                sub_pattern INTEGER NOT NULL,
                count INTEGER,
                success INTEGER,
                PRIMARY KEY (level2_pattern, sub_pattern)
            );
            """
        )
        conn.commit()
        conn.close()

    def deleteLevel2RiskPatternBySubPattern(self, sub_pattern: int) -> None:

        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM level2RiskPattern
            WHERE sub_pattern = ?;
            """,
            (sub_pattern,),
        )
        conn.commit()
        conn.close()

    def addLevel2RiskPattern(
        self,
        level2_pattern: int,
        risk_pattern: str,
        action_id: int,
        dont_action_id_1: Optional[int],
        dont_action_id_2: Optional[int],
        dont_action_id_3: Optional[int],
        sub_pattern: int,
    ) -> None:


        conn = sqlite3.connect(self.database)
        cur = conn.cursor()

        ts_now = (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat(timespec="seconds")

 
        cur.execute(
            """
            SELECT count, risk_pattern FROM level2RiskPattern
            WHERE level2_pattern = ? AND sub_pattern = ?;
            """,
            (level2_pattern, sub_pattern),
        )
        row = cur.fetchone()

        if row:

            current_count = row[0]  
            new_count = current_count + 1  
            existing_risk_pattern = row[1]  
            updated_risk_pattern = f"{existing_risk_pattern}, {risk_pattern}" if existing_risk_pattern else risk_pattern
            cur.execute(
                """
                UPDATE level2RiskPattern
                SET risk_pattern = ?, action_id = ?, 
                    dont_action_id_1 = ?, dont_action_id_2 = ?, dont_action_id_3 = ?, 
                    ts_last = ?, count = ?
                WHERE level2_pattern = ? AND sub_pattern = ?;
                """,
                (
                    updated_risk_pattern,
                    action_id,
                    dont_action_id_1,
                    dont_action_id_2,
                    dont_action_id_3,
                    ts_now,
                    new_count,
                    level2_pattern,
                    sub_pattern,
                ),
            )
        else:
 
            cur.execute(
                """
                INSERT INTO level2RiskPattern (
                    level2_pattern, risk_pattern, action_id, 
                    dont_action_id_1, dont_action_id_2, dont_action_id_3, 
                    ts_first, ts_last, sub_pattern, count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    level2_pattern,
                    risk_pattern,
                    action_id,
                    dont_action_id_1,
                    dont_action_id_2,
                    dont_action_id_3,
                    ts_now,
                    ts_now,
                    sub_pattern,
                    1,  
                ),
            )
        

        conn.commit()
        conn.close()


    def getLevel2RiskPattern2(
        self, 
        level2_pattern1: int, 
        sub_pattern1: int, 
        level2_pattern2: int, 
        sub_pattern2: int
    ) -> List[Dict[str, Any]]:

        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM level2RiskPattern
            WHERE (Level2_pattern = ? AND Sub_pattern = ?)
            OR (Level2_pattern = ? AND Sub_pattern = ?);
            """,
            (level2_pattern1, sub_pattern1, level2_pattern2, sub_pattern2),
        )
        rows = cur.fetchall()
        conn.close()

        result = []
        for row in rows:
            result.append({
                "level2_pattern": row[0],
                "risk_pattern": row[1],
                "action_id": row[2],
                "dont_action_id_1": row[3],
                "dont_action_id_2": row[4],
                "dont_action_id_3": row[5],
                "ts_first": row[6],
                "ts_last": row[7],
                "sub_pattern": row[8],
                "count": row[9],
                "success": row[10],
            })

        return result

    def getLevel2RiskPattern(self, level2_pattern: int, sub_pattern: int) -> Optional[Dict[str, Any]]:

        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM level2RiskPattern
            WHERE Level2_pattern = ? AND Sub_pattern = ?;
            """,
            (level2_pattern, sub_pattern),
        )
        row = cur.fetchone()
        conn.close()

        if row:
            return {
                "level2_pattern": row[0],
                "risk_pattern": row[1],
                "action_id": row[2],
                "dont_action_id_1": row[3],
                "dont_action_id_2": row[4],
                "dont_action_id_3": row[5],
                "ts_first": row[6],
                "ts_last": row[7],
                "sub_pattern": row[8],
                "count": row[9],
                "success": row[10],
            }
        return None

    def updateTsLast(self, level2_pattern: int, sub_pattern: int) -> None:

        ts_now = (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat(timespec="seconds")
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE level2RiskPattern
            SET ts_last = ?
            WHERE level2_pattern = ? AND sub_pattern = ?;
            """,
            (ts_now, level2_pattern, sub_pattern),
        )
        conn.commit()
        conn.close()



    def should_not_do_which_action(
        self,
        risk_pattern: List[int],
        current_available_actions_str: str,
    ) -> Tuple[bool, str, List[int]]:


        is_updated = False
        removed_actions = []


        level2_pattern1 = int(f"8{risk_pattern[1]}{risk_pattern[3]}{risk_pattern[4]}{risk_pattern[5]}")
        level2_pattern2 = int(f"2{risk_pattern[13]}{risk_pattern[9]}{risk_pattern[10]}{risk_pattern[11]}")

        matched_records = self.getLevel2RiskPattern2(
            level2_pattern1=level2_pattern1,
            sub_pattern1=0,  # Turn-left
            level2_pattern2=level2_pattern2,
            sub_pattern2=2,  # Turn-right
        )


        if not matched_records:
            return is_updated, current_available_actions_str, removed_actions


        dont_action_ids = []
        for record in matched_records:
            dont_action_ids.extend([
                record.get("dont_action_id_1"),
                record.get("dont_action_id_2"),
                record.get("dont_action_id_3"),
            ])

        dont_action_ids = list(set(action_id for action_id in dont_action_ids if action_id is not None and 0 <= action_id <= 4))


        if dont_action_ids:
            is_updated = True
            removed_actions = dont_action_ids


            result_str = remove_actions_from_string(current_available_actions_str, dont_action_ids)
        else:
            result_str = current_available_actions_str

        return is_updated, result_str, removed_actions
    
    def should_do_change_lane(
        self,
        risk_pattern: List[int],
        sub_pattern: int,
    ) -> Tuple[bool, Optional[int]]:

        is_matched = False


        if sub_pattern == 3:  # Front risk high situation
            level2_pattern = int(f"3{risk_pattern[7]}{risk_pattern[8]}")
        elif sub_pattern == 4:  # Behind risk high situation
            level2_pattern = int(f"4{risk_pattern[7]}{risk_pattern[6]}")
        else:

            return is_matched

        record = self.getLevel2RiskPattern(level2_pattern=level2_pattern, sub_pattern=sub_pattern)


        if record:
            is_matched = True

        return is_matched


    def sporty_action(
        self,
        risk_pattern: List[int],
        sub_pattern: int,
    ) -> Tuple[bool, Optional[int]]:

        is_matched = False

        if sub_pattern == 9:  # prefer acceleration situation
            level2_pattern = int(f"9{risk_pattern[8]}")
        elif sub_pattern == 5:  # turn left for future acceleration situation
            level2_pattern = int(f"5{risk_pattern[4]}{risk_pattern[5]}")
        elif sub_pattern == 6:  # turn right for future acceleration situation
            level2_pattern = int(f"6{risk_pattern[10]}{risk_pattern[11]}")
        elif sub_pattern == 10:  # prefer deceleration situation
            level2_pattern = int(f"10{risk_pattern[8]}")
        else:

            print(f"[WARNING] sub_pattern {sub_pattern} is not supported for sporty action.")
            return is_matched


        record = self.getLevel2RiskPattern(level2_pattern=level2_pattern, sub_pattern=sub_pattern)


        if record:
            is_matched = True

        return is_matched


    def should_do_which_action(
        self,
        risk_pattern: List[int],
        sub_pattern: int,
    ) -> Tuple[bool, Optional[int]]:

        is_matched = False
        matched_action_id = None


        if sub_pattern == 3:  # Front risk high situation
            level2_pattern = int(f"3{risk_pattern[7]}{risk_pattern[8]}")
        elif sub_pattern == 4:  # Behind risk high situation
            level2_pattern = int(f"4{risk_pattern[7]}{risk_pattern[6]}")
        else:

            return is_matched, matched_action_id


        record = self.getLevel2RiskPattern(level2_pattern=level2_pattern, sub_pattern=sub_pattern)


        if record:
            is_matched = True
            matched_action_id = record.get("action_id")

        return is_matched, matched_action_id

    def print_record_count(self) -> None:


        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM level2RiskPattern;")
        count = cur.fetchone()[0]
        conn.close()
        print(f"[INFO] Total records in level2RiskPattern: {count}")

    def print_all_records_with_index(self) -> None:

        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute("SELECT * FROM level2RiskPattern;")
        rows = cur.fetchall()
        conn.close()

        if rows:
            print("[INFO] All records in level2RiskPattern:")
            for index, row in enumerate(rows, start=1):
                print(f"[Record {index}] {row}")
        else:
            print("[INFO] No records found in level2RiskPattern.")