import lief
import array
import hashlib
import numpy as np
import random
import os
import sys


module_path = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]

COMMON_SECTION_NAMES = open(os.path.join(
    module_path, 'section_names.txt'), 'r').read().rstrip().split('\n')


class PEModifier(object):
    def __init__(self, exe_filepath):
        self.sha256 = None
        self.exe_filepath = exe_filepath
        self.lief_errors = (lief.bad_format, lief.bad_file, lief.pe_error, lief.parser_error, lief.read_out_of_bound,
               RuntimeError)
        self.bytez, self.bytez_int_list, self.lief_binary = self._is_valid(self.exe_filepath)

    def _is_valid(self, exe_filepath):
        bytez = self._get_bytez(exe_filepath)
        bytez_int_list = self._convert_to_int(bytez)
        # Check that before the modifications the file is valid!
        lief_binary = self._get_binary(bytez_int_list)
        return bytez, bytez_int_list, lief_binary

    def _get_binary(self, bytez_int_list):
        try:
            binary = lief.PE.parse(bytez_int_list)
        except self.lief_errors as e:
            print("lief error: ", str(e))
            binary = None
            raise
        except Exception:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
            raise
        return binary

    def _get_bytez(self, exe_filepath):
        with open(exe_filepath, "rb") as f:
            bytez = f.read()
            self.sha256 = hashlib.sha256(bytez).hexdigest()
        return bytez

    def _convert_to_int(self, bytez):
        return list(bytez)

    def overlay_append(self, overlay_int_list):
        self.bytez_int_list = self.bytez_int_list + overlay_int_list
        return self.bytez_int_list

    def _build_binary(self, binary, dos_stub=False, imports=False, overlay=False, relocations=False,
                          resources=False, tls=False):
        builder = lief.PE.Builder(binary)  # write the file back as bytez
        if (dos_stub):
            builder.build_dos_stub(dos_stub)  # rebuild DOS stub
        if (imports):
            builder.build_imports(imports)  # rebuild IAT in another section
            builder.patch_imports(imports)  # patch orig. import table with trampolines to new import table
        if (overlay):
            builder.build_overlay(overlay)  # rebuild overlay
        if (relocations):
            builder.build_relocations(relocations)  # rebuild relocation table in another section
        if (resources):
            builder.build_resources(resources)  # rebuild resources in another section
        if (tls):
            builder.build_tls(tls)  # rebuilt TLS object in another section
        builder.build()  # perform the build process
        return builder

    def _binary_to_bytez(self, binary, dos_stub=False, imports=False, overlay=False, relocations=False,
                          resources=False, tls=False):
        self.lief_binary = binary
        self.builder = self._build_binary(
            self.lief_binary,
            dos_stub=dos_stub,
            imports=imports,
            overlay=overlay,
            relocations=relocations,
            resources=resources,
            tls=tls
        )
        self.bytez_int_list = self.builder.get_build()
        self.bytez = array.array('B', self.bytez_int_list).tobytes()
        return self.bytez, self.bytez_int_list, self.lief_binary

    def add_import(self, library_name, function_name):
        """ Add a function of a given library to the Import Address Table"""
        lowerlibname = library_name.lower()
        # find this lib in the imports, if it exists
        lib = [im for im in self.lief_binary.imports if im.name.lower() == lowerlibname]
        if lib == []:
            lib = self.lief_binary.add_library(library_name)
        else:
            lib = lib[0]

        # get current names
        names = {e.name for e in lib.entries}
        if function_name not in names:
            lib.add_entry(function_name)

        # Remember to call the following lines
        #self.bytez, self.bytez_int_list, self.lief_binary = self._binary_to_bytez(self.lief_binary, imports=True)
        #return self.bytez, self.bytez_int_list, self.lief_binary

    def add_imports(self, import_features, inverse_vocabulary_mapping):
        for j, val in enumerate(import_features):
            if val == 1.0:
                library_and_function_names = inverse_vocabulary_mapping[str(j)]
                library_name = library_and_function_names.split(";")[0]
                function_name = library_and_function_names.split(";")[1]

                lowerlibname = library_name.lower()
                # find this lib in the imports, if it exists
                lib = [im for im in self.lief_binary.imports if im.name.lower() == lowerlibname]
                if lib == []:
                    lib = self.lief_binary.add_library(library_name)
                else:
                    lib = lib[0]

                # get current names
                names = {e.name for e in lib.entries}
                if function_name not in names:
                    lib.add_entry(function_name)

        self.bytez, self.bytez_int_list, self.lief_binary = self._binary_to_bytez(
            self.lief_binary,
            imports=True
        )
        return self.bytez, self.bytez_int_list, self.lief_binary

    def strings_to_content(self, strings_to_inject: list):
        new_content = []
        for string_to_inject in strings_to_inject:
            new_content += list(str.encode(string_to_inject))
        return new_content

    def create_new_section(
            self,
            section_content: list,
            section_name: str=None):
        """
        Steps:
        1/ Obtain new strings you need to inject
        2/ Obtain a section name
        3/ Create new section
        4/ Inject the strings into the section's content
        5/ Build PE
        """
        if section_name is None:
            section_names = [section.name for section in self.lief_binary.sections]
            targeted_section_name = random.choice(COMMON_SECTION_NAMES)[:7]
            while targeted_section_name in section_names:
                targeted_section_name = random.choice(COMMON_SECTION_NAMES)[:7]
        else:
            targeted_section_name = section_name

        print("Section name: ", targeted_section_name)

        new_section = lief.PE.Section(targeted_section_name)
        new_section.content = section_content
        self.lief_binary.add_section(new_section,
                           random.choice([
                               lief.PE.SECTION_TYPES.BSS,
                               lief.PE.SECTION_TYPES.DATA,
                               lief.PE.SECTION_TYPES.EXPORT,
                               lief.PE.SECTION_TYPES.IDATA,
                               lief.PE.SECTION_TYPES.RELOCATION,
                               lief.PE.SECTION_TYPES.RESOURCE,
                               lief.PE.SECTION_TYPES.TEXT,
                               lief.PE.SECTION_TYPES.TLS_,
                               lief.PE.SECTION_TYPES.UNKNOWN,
                           ]))
        self.bytez, self.bytez_int_list, self.lief_binary = self._binary_to_bytez(
            self.lief_binary,
        )
        return self.bytez, self.bytez_int_list, self.lief_binary
