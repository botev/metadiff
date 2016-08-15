//
// Created by alex on 19/03/15.
//

#ifndef METADIFF_METADIFF_H
#define METADIFF_METADIFF_H

#include <memory>
#include <iostream>
#include <iomanip>
#include <exception>
#include <fstream>
#include <dlfcn.h>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_set>
#include <unordered_map>

using std::vector;
using std::string;
using std::unordered_set;
using std::unordered_map;
using std::shared_ptr;
using std::map;
using std::ios;

#include "os.h"
#include "logging.h"
#include "symbolic.h"
#include "defs.h"
#include "shared.h"
#include "core.h"
#include "exceptions.h"
#include "core_impl.h"
#include "operators.h"
#include "visual.h"
#include "backends.h"
#include "api.h"
#include "optimize.h"


#endif //METADIFF_METADIFF_H