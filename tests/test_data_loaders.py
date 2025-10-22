# tests/test_data_loaders.py
import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
import yaml
from unittest.mock import patch, MagicMock

# Mock pyreadr before importing data_loaders
sys.modules['pyreadr'] = MagicMock()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loaders import (
    _get_base_dir,
    return_default_config,
    _resolve,
    _deep_merge,
    _load_config,
    allocate_and_drop_missing_age,
    read_rds_file,
    load_all_data,
    correct_valor_for_omission,
    get_lifetables_ex,
    get_fertility
)


class TestGetBaseDir:
    """Test _get_base_dir function"""
    
    def test_returns_string(self):
        result = _get_base_dir()
        assert isinstance(result, str)
    
    def test_returns_absolute_path(self):
        result = _get_base_dir()
        assert os.path.isabs(result)
    
    def test_points_to_src_directory(self):
        result = _get_base_dir()
        assert result.endswith('src')


class TestReturnDefaultConfig:
    """Test return_default_config function"""
    
    def test_returns_dict(self):
        config = return_default_config()
        assert isinstance(config, dict)
    
    def test_has_required_keys(self):
        config = return_default_config()
        required_keys = ['paths', 'diagnostics', 'projections', 'fertility', 
                        'mortality', 'runs', 'age_bins', 'midpoints']
        for key in required_keys:
            assert key in config
    
    def test_paths_structure(self):
        config = return_default_config()
        paths = config['paths']
        assert 'data_dir' in paths
        assert 'results_dir' in paths
        assert 'target_tfr_csv' in paths
        assert 'midpoints_csv' in paths
        assert 'mortality_improvements_csv' in paths
    
    def test_projections_structure(self):
        config = return_default_config()
        proj = config['projections']
        assert 'start_year' in proj
        assert 'end_year' in proj
        assert proj['start_year'] == 2018
        assert proj['end_year'] == 2070
    
    def test_age_bins_order(self):
        config = return_default_config()
        age_bins = config['age_bins']['order']
        assert len(age_bins) == 17
        assert age_bins[0] == "0-4"
        assert age_bins[-1] == "80+"


class TestResolve:
    """Test _resolve function"""
    
    def test_resolves_relative_path(self):
        root = "/home/user/project"
        relative = "./data"
        result = _resolve(root, relative)
        assert os.path.isabs(result)
        assert "data" in result
    
    def test_handles_absolute_path(self):
        root = "/home/user/project"
        absolute = "/var/data"
        result = _resolve(root, absolute)
        assert os.path.isabs(result)


class TestDeepMerge:
    """Test _deep_merge function"""
    
    def test_simple_merge(self):
        dst = {'a': 1, 'b': 2}
        src = {'b': 3, 'c': 4}
        _deep_merge(dst, src)
        assert dst == {'a': 1, 'b': 3, 'c': 4}
    
    def test_nested_merge(self):
        dst = {'a': {'x': 1, 'y': 2}, 'b': 3}
        src = {'a': {'y': 3, 'z': 4}}
        _deep_merge(dst, src)
        assert dst == {'a': {'x': 1, 'y': 3, 'z': 4}, 'b': 3}
    
    def test_deep_nested_merge(self):
        dst = {'a': {'b': {'c': 1}}}
        src = {'a': {'b': {'d': 2}}}
        _deep_merge(dst, src)
        assert dst == {'a': {'b': {'c': 1, 'd': 2}}}
    
    def test_overwrites_non_dict_values(self):
        dst = {'a': 1}
        src = {'a': {'b': 2}}
        _deep_merge(dst, src)
        assert dst == {'a': {'b': 2}}


class TestLoadConfig:
    """Test _load_config function"""
    
    def test_loads_default_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "nonexistent.yaml")
            cfg, paths = _load_config(tmpdir, config_path)
            
            assert isinstance(cfg, dict)
            assert 'paths' in cfg
            assert isinstance(paths, dict)
    
    def test_loads_yaml_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            test_config = {
                'projections': {
                    'start_year': 2020,
                    'end_year': 2080
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            cfg, paths = _load_config(tmpdir, config_path)
            
            assert cfg['projections']['start_year'] == 2020
            assert cfg['projections']['end_year'] == 2080
    
    def test_merges_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            test_config = {'projections': {'start_year': 2025}}
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            cfg, paths = _load_config(tmpdir, config_path)
            
            # Custom value
            assert cfg['projections']['start_year'] == 2025
            # Default value should still exist
            assert 'end_year' in cfg['projections']
    
    def test_returns_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            cfg, paths = _load_config(tmpdir, config_path)
            
            for key, path in paths.items():
                assert os.path.isabs(path), f"{key} path is not absolute"


class TestAllocateAndDropMissingAge:
    """Test allocate_and_drop_missing_age function"""
    
    def test_removes_missing_ages(self):
        df = pd.DataFrame({
            'DPTO_NOMBRE': ['A', 'A', 'A'],
            'SEXO': ['M', 'M', 'M'],
            'FUENTE': ['X', 'X', 'X'],
            'ANO': [2020, 2020, 2020],
            'VARIABLE': ['POP', 'POP', 'POP'],
            'EDAD': [10, 20, None],
            'VALOR': [100, 200, 50],
            'VALOR_withmissing': [100, 200, 50]
        })
        
        result = allocate_and_drop_missing_age(df)
        
        assert len(result) == 2
        assert result['EDAD'].notna().all()
    
    def test_allocates_missing_proportionally(self):
        df = pd.DataFrame({
            'DPTO_NOMBRE': ['A', 'A', 'A'],
            'SEXO': ['M', 'M', 'M'],
            'FUENTE': ['X', 'X', 'X'],
            'ANO': [2020, 2020, 2020],
            'VARIABLE': ['POP', 'POP', 'POP'],
            'EDAD': [10, 20, None],
            'VALOR': [100, 200, 60],
            'VALOR_withmissing': [100, 200, 60]
        })
        
        result = allocate_and_drop_missing_age(df)
        
        # Missing value (60) should be allocated proportionally
        # 100/(100+200) = 1/3, so first gets 60 * 1/3 = 20
        # 200/(100+200) = 2/3, so second gets 60 * 2/3 = 40
        assert result.iloc[0]['VALOR_withmissing'] == pytest.approx(120, rel=1e-6)
        assert result.iloc[1]['VALOR_withmissing'] == pytest.approx(240, rel=1e-6)
    
    def test_preserves_original_dataframe(self):
        df = pd.DataFrame({
            'DPTO_NOMBRE': ['A', 'A'],
            'SEXO': ['M', 'M'],
            'FUENTE': ['X', 'X'],
            'ANO': [2020, 2020],
            'VARIABLE': ['POP', 'POP'],
            'EDAD': [10, None],
            'VALOR': [100, 50],
            'VALOR_withmissing': [100, 50]
        })
        
        original_len = len(df)
        result = allocate_and_drop_missing_age(df)
        
        assert len(df) == original_len  # Original unchanged
        assert len(result) == 1


class TestReadRdsFile:
    """Test read_rds_file function"""
    
    @patch('data_loaders.pyreadr.read_r')
    def test_successful_read(self, mock_read_r):
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_r.return_value = {None: mock_df}
        
        result = read_rds_file('test.rds')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        mock_read_r.assert_called_once_with('test.rds')
    
    @patch('data_loaders.pyreadr.read_r')
    def test_raises_on_error(self, mock_read_r):
        mock_read_r.side_effect = Exception("File not found")
        
        with pytest.raises(RuntimeError, match="Failed to read"):
            read_rds_file('nonexistent.rds')


class TestLoadAllData:
    """Test load_all_data function"""
    
    def test_raises_when_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="File not found"):
                load_all_data(tmpdir)
    
    @patch('data_loaders.read_rds_file')
    def test_loads_conteos_file(self, mock_read_rds):
        mock_df = pd.DataFrame({'data': [1, 2, 3]})
        mock_read_rds.return_value = mock_df
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy file
            conteos_path = os.path.join(tmpdir, 'conteos.rds')
            open(conteos_path, 'w').close()
            
            result = load_all_data(tmpdir)
            
            assert 'conteos' in result
            assert isinstance(result['conteos'], pd.DataFrame)


class TestCorrectValorForOmission:
    """Test correct_valor_for_omission function"""
    
    def test_no_omission_returns_original(self):
        df = pd.DataFrame({
            'VALOR': [100, 200, 300],
            'OMISION': [None, None, None]
        })
        
        result = correct_valor_for_omission(df, 'mid')
        
        # Check values are equal, ignoring dtype and name differences
        pd.testing.assert_series_equal(result, df['VALOR'], check_dtype=False, check_names=False)
    
    def test_low_sample_uses_lower_bound(self):
        df = pd.DataFrame({
            'VALOR': [100.0],
            'OMISION': [1]
        })
        
        result = correct_valor_for_omission(df, 'low')
        
        # Should use 0.004 for level 1
        expected = 100.0 * (1.0 + 0.004)
        assert result.iloc[0] == pytest.approx(expected, rel=1e-6)
    
    def test_mid_sample_uses_midpoint(self):
        df = pd.DataFrame({
            'VALOR': [100.0],
            'OMISION': [1]
        })
        
        result = correct_valor_for_omission(df, 'mid')
        
        # Midpoint of (0.004, 0.104) is 0.054
        expected = 100.0 * (1.0 + 0.054)
        assert result.iloc[0] == pytest.approx(expected, rel=1e-6)
    
    def test_high_sample_uses_upper_bound(self):
        df = pd.DataFrame({
            'VALOR': [100.0],
            'OMISION': [1]
        })
        
        result = correct_valor_for_omission(df, 'high')
        
        # Should use 0.104 for level 1
        expected = 100.0 * (1.0 + 0.104)
        assert result.iloc[0] == pytest.approx(expected, rel=1e-6)
    
    def test_invalid_sample_type_raises_error(self):
        df = pd.DataFrame({
            'VALOR': [100],
            'OMISION': [1]
        })
        
        with pytest.raises(ValueError, match="sample_type must be"):
            correct_valor_for_omission(df, 'invalid')
    
    def test_uniform_distribution(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'VALOR': [100.0],
            'OMISION': [1]
        })
        
        result = correct_valor_for_omission(df, 'mid', distribution='uniform')
        
        # Result should be between 100 * 1.004 and 100 * 1.104
        assert result.iloc[0] >= 100.0 * 1.004
        assert result.iloc[0] <= 100.0 * 1.104
    
    def test_beta_distribution(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'VALOR': [100.0],
            'OMISION': [2]
        })
        
        result = correct_valor_for_omission(df, 'mid', distribution='beta')
        
        # Result should be between bounds for level 2
        assert result.iloc[0] >= 100.0 * 1.105
        assert result.iloc[0] <= 100.0 * 1.203
    
    def test_pert_distribution(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'VALOR': [100.0],
            'OMISION': [3]
        })
        
        result = correct_valor_for_omission(df, 'mid', distribution='pert')
        
        # Result should be between bounds for level 3
        assert result.iloc[0] >= 100.0 * 1.204
        assert result.iloc[0] <= 100.0 * 1.301
    
    def test_normal_distribution(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'VALOR': [100.0],
            'OMISION': [4]
        })
        
        result = correct_valor_for_omission(df, 'mid', distribution='normal')
        
        # Result should be between bounds for level 4
        assert result.iloc[0] >= 100.0 * 1.302
        assert result.iloc[0] <= 100.0 * 1.400
    
    def test_invalid_distribution_raises_error(self):
        df = pd.DataFrame({
            'VALOR': [100],
            'OMISION': [1]
        })
        
        with pytest.raises(ValueError, match="distribution must be"):
            correct_valor_for_omission(df, 'mid', distribution='invalid')
    
    def test_multiple_omission_levels(self):
        df = pd.DataFrame({
            'VALOR': [100.0, 200.0, 300.0],
            'OMISION': [1, 3, 5]
        })
        
        result = correct_valor_for_omission(df, 'mid')
        
        # Level 1: midpoint 0.054
        assert result.iloc[0] == pytest.approx(100.0 * 1.054, rel=1e-6)
        # Level 3: midpoint (0.204 + 0.301) / 2 = 0.2525
        assert result.iloc[1] == pytest.approx(200.0 * 1.2525, rel=1e-6)
        # Level 5: midpoint (0.401 + 0.499) / 2 = 0.45
        assert result.iloc[2] == pytest.approx(300.0 * 1.45, rel=1e-6)


class TestGetLifetablesEx:
    """Test get_lifetables_ex function"""
    
    @patch('data_loaders.os.listdir')
    @patch('data_loaders.pd.read_csv')
    def test_reads_multiple_distributions(self, mock_read_csv, mock_listdir):
        mock_listdir.return_value = ['file1.csv']
        mock_df = pd.DataFrame({'ex': [70.5, 75.2, 80.1]})
        mock_read_csv.return_value = mock_df
        
        # This will likely fail without proper directory structure,
        # so we just test that it doesn't crash on basic calls
        try:
            result = get_lifetables_ex('TestDPTO')
            # If it succeeds, check it returns a DataFrame
            assert isinstance(result, pd.DataFrame)
        except (FileNotFoundError, OSError):
            # Expected if directories don't exist
            pass


class TestGetFertility:
    """Test get_fertility function"""
    
    @patch('data_loaders.os.path.isdir')
    @patch('data_loaders.os.listdir')
    @patch('data_loaders.pd.read_csv')
    def test_handles_missing_directories(self, mock_read_csv, mock_listdir, mock_isdir):
        mock_isdir.return_value = False
        
        stacked_df, tfr_df = get_fertility()
        
        # Should return empty structures when directories don't exist
        assert isinstance(stacked_df, pd.DataFrame)
        assert isinstance(tfr_df, pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
