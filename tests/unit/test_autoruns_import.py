import unittest

from src.web.autoruns_import import parse_autoruns_text


class AutorunsImportParserTests(unittest.TestCase):
    def test_parse_csv(self):
        payload = (
            "Entry,Entry Location,Enabled,Category,Profile,Description,Publisher,Image Path,Launch String,Signer,Verified,VirusTotal\n"
            "OneDrive,HKCU\\\\Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run,Enabled,Logon,user,OneDrive startup,Microsoft Corporation,C:\\\\Program Files\\\\Microsoft OneDrive\\\\OneDrive.exe,\\\"C:\\\\Program Files\\\\Microsoft OneDrive\\\\OneDrive.exe\\\",Microsoft Corporation,Signed,0/74\n"
            "UnknownTask,Task Scheduler,Enabled,Scheduled Tasks,user,Unknown task,Unknown,C:\\\\Temp\\\\unknown.exe,C:\\\\Temp\\\\unknown.exe,,Not verified,6/74\n"
        )
        rows = parse_autoruns_text(payload)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["entry_name"], "OneDrive")
        self.assertEqual(rows[1]["vt_positives"], 6)
        self.assertEqual(rows[1]["vt_total"], 74)

    def test_parse_tsv(self):
        payload = (
            "Entry\tEntry Location\tEnabled\tImage Path\tLaunch String\tVirusTotal\n"
            "Updater\tHKLM\\\\Run\tEnabled\tC:\\\\app\\\\updater.exe\tC:\\\\app\\\\updater.exe\t0/72\n"
        )
        rows = parse_autoruns_text(payload)
        self.assertEqual(len(rows), 1)
        self.assertIn("HKLM", rows[0]["entry_location"])
        self.assertIn("Run", rows[0]["entry_location"])
        self.assertEqual(rows[0]["enabled"], True)


if __name__ == "__main__":
    unittest.main()
