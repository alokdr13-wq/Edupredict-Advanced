import unittest
from auth.authentication import hash_password, verify_password
from utils.helpers import recommendations

class TestEduPredict(unittest.TestCase):
    def test_password_hash_round_trip(self):
        encoded=hash_password("VeryStrongPass123!")
        self.assertTrue(verify_password("VeryStrongPass123!",encoded))
        self.assertFalse(verify_password("wrong-password",encoded))

    def test_recommendations_are_nonempty(self):
        tips=recommendations({"G1":5,"G2":7,"studytime":1,"absences":15,"failures":1,"schoolsup":"no"},8.0)
        self.assertTrue(len(tips)>=3)

    def test_recommendations_have_default(self):
        tips=recommendations({"G1":15,"G2":16,"studytime":3,"absences":0,"failures":0,"schoolsup":"yes"},16.0)
        self.assertGreaterEqual(len(tips),1)

if __name__=="__main__":
    unittest.main()
