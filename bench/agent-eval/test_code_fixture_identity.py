import unittest
from code_fixture import validate

CORRECT = '''def invoice_total(rows):
    total = 0
    for row in rows:
        quantity = row['quantity']
        price = row['unit_price_cents']
        discount = row.get('discount_percent', 0)
        if discount is None:
            discount = 0
        if quantity < 0 or price < 0 or discount < 0 or discount > 100:
            raise ValueError()
        total += quantity * price * (100-discount) // 100
    return total
'''


class IdentityTest(unittest.TestCase):
    def test_none_identity_allows_behavioral_evaluation(self):
        self.assertTrue(validate(CORRECT)['passed'])
        self.assertTrue(validate(CORRECT.replace('discount is None','None is discount'))['passed'])
        self.assertTrue(validate(CORRECT.replace('if discount is None:\n            discount = 0',
                                               'if discount is not None:\n            discount = discount'))['passed'])

    def test_identity_does_not_allow_other_object_comparisons(self):
        result=validate(CORRECT.replace('discount is None','discount is rows'))
        self.assertIsNone(result['passed'])
        self.assertEqual(result['reason'],'restricted_identity')

    def test_rounding_bug_remains_a_failed_test(self):
        result=validate(CORRECT.replace('quantity * price * (100-discount) // 100',
                                       'quantity * price - quantity * price * discount // 100'))
        self.assertFalse(result['passed'])
        self.assertEqual(result['cases'],[True,False,False,True,True,True,True,True])


if __name__=='__main__':unittest.main()
