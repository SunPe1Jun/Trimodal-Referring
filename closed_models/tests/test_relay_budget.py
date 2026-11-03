import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('tested_budget',Path(__file__).resolve().parents[1]/'relay_budget.py')
budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)


def usage(balance,actual=0.,account=0.,nominal=0.):
    return {'unit':'USD','balance':balance,'model_stats':[{'actual_cost':actual,'account_cost':account,'cost':nominal}]}


class WalletTests(unittest.TestCase):
    def test_nominal_cost_does_not_get_charged_again_as_wallet_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(budget,'STATE',Path(directory)/'state.json'):
                with patch.object(budget,'snapshot',return_value=usage(70.,1.,3.,10.)):
                    budget.initialize({},300.,10.)
                with patch.object(budget,'snapshot',return_value=usage(68.,3.,5.,100.)):
                    status,balance,spent=budget.check({},4)
                self.assertEqual(status,'RUNNING')
                self.assertEqual(spent,2.)
                self.assertEqual(balance,68.)

    def test_low_wallet_waits_without_claiming_funds_already_added(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(budget,'STATE',Path(directory)/'state.json'):
                with patch.object(budget,'snapshot',return_value=usage(11.)):
                    budget.initialize({},300.,10.)
                    status,_,_=budget.check({},32)
                self.assertEqual(status,'WAITING_TOPUP')

    def test_topup_can_resume_and_nominal_discount_is_not_applied_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(budget,'STATE',Path(directory)/'state.json'):
                with patch.object(budget,'snapshot',return_value=usage(11.)):
                    budget.initialize({},300.,10.)
                with patch.object(budget,'snapshot',return_value=usage(111.)):
                    status,_,spent=budget.check({},32)
                self.assertEqual(status,'RUNNING')
                self.assertEqual(spent,0.)

    def test_run_cap_is_shared_and_reservations_are_released(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(budget,'STATE',Path(directory)/'state.json'):
                with patch.object(budget,'snapshot',return_value=usage(100.)):
                    budget.initialize({},5.,10.)
                    self.assertEqual(budget.check({},32)[0],'CAP_REACHED')
                    self.assertEqual(budget.check({},4)[0],'RUNNING')
                    budget.release()
                    self.assertEqual(json.loads(budget.STATE.read_text())['batch_reservations'],{})


if __name__=='__main__':unittest.main()
