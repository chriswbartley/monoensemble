import numpy
from numpy.distutils.misc_util import Configuration

# REMEMBER: when running manually, include run options
# build_ext --inplace





def configuration(parent_package="", top_path=None):
    config = Configuration("monoensemble", parent_package, top_path)
    pyx_file='monoensemble/_mono_gradient_boosting.pyx'
    c_file='monoensemble/_mono_gradient_boosting.c'
    if os.path.isfile(pyx_file):
        ########################
        # START Monkey patch code to get cython to work #
        # https://stackoverflow.com/questions/37178055/attributeerror-list-object-has-no-attribute-rfind-using-petsc4py #
        ########################
        
        from numpy.distutils.misc_util import appendpath
        from numpy.distutils import log
        from os.path import join as pjoin, dirname
        from distutils.dep_util import newer_group
        from distutils.errors import DistutilsError
        
        
        # $ python setup.py build_ext --inplace
        
        from numpy.distutils.command import build_src
        
        # a bit of monkeypatching ...
        import Cython.Compiler.Main
        build_src.Pyrex = Cython
        build_src.have_pyrex = True
        
        
        def have_pyrex():
            import sys
            try:
                import Cython.Compiler.Main
                sys.modules['Pyrex'] = Cython
                sys.modules['Pyrex.Compiler'] = Cython.Compiler
                sys.modules['Pyrex.Compiler.Main'] = Cython.Compiler.Main
                return True
            except ImportError:
                return False
        build_src.have_pyrex = have_pyrex
        
        def generate_a_pyrex_source(self, base, ext_name, source, extension):
            ''' Monkey patch for numpy build_src.build_src method
            Uses Cython instead of Pyrex.
            Assumes Cython is present
            '''
            if self.inplace:
                target_dir = dirname(base)
            else:
                target_dir = appendpath(self.build_src, dirname(base))
            target_file = pjoin(target_dir, ext_name + '.c')
            depends = [source] + extension.depends
            if self.force or newer_group(depends, target_file, 'newer'):
                import Cython.Compiler.Main
                log.info("cythonc:> %s" % (target_file))
                self.mkpath(target_dir)
                options = Cython.Compiler.Main.CompilationOptions(
                    defaults=Cython.Compiler.Main.default_options,
                    include_path=extension.include_dirs,
                    output_file=target_file)
                cython_result = Cython.Compiler.Main.compile(source, options=options)
                if cython_result.num_errors != 0:
                    raise DistutilsError("%d errors while compiling %r with Cython" % (cython_result.num_errors, source))
            return target_file
        
        build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source
        
        ########################
        # END additional code  #
        ########################
                
        config.add_extension("_mono_gradient_boosting",
                             sources=["monoensemble/_mono_gradient_boosting.pyx"],
                             include_dirs=['.', numpy.get_include()])
    else:
        config.add_extension("_mono_gradient_boosting",
                             sources=["monoensemble/_mono_gradient_boosting.c"],
                             include_dirs=['.', numpy.get_include()])
        
    #config.add_subpackage("tests")

    return config

import os
from setuptools import find_packages
PACKAGES = find_packages()

# Get version and release info, which is all stored in monoensemble/version.py
ver_file = os.path.join('monoensemble', 'version.py')
with open(ver_file) as f:
    exec(f.read())
    
opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            install_requires=INSTALL_REQUIRES,
            requires=REQUIRES,
            zip_safe = False)

if __name__ == "__main__":
    from numpy.distutils.core import setup
    config_dict=configuration().todict()
    for key in config_dict.keys():
        opts[key]=config_dict[key]
    setup(**opts)
    
    

    
    