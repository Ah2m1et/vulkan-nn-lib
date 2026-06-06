if(NOT Vulkan_FOUND)
    return()
endif()

# Prefer glslc; fall back to glslangValidator
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin)
if(NOT GLSLC)
    find_program(GLSLC glslangValidator)
    if(GLSLC)
        message(STATUS "glslc not found — using glslangValidator for shader compilation.")
    else()
        message(WARNING "Neither glslc nor glslangValidator found — shaders will not be compiled.")
        return()
    endif()
endif()

# Detect whether the resolved binary is glslangValidator (needs -V flag for SPIR-V output).
get_filename_component(_GLSLC_NAME "${GLSLC}" NAME)
if(_GLSLC_NAME MATCHES "glslangValidator")
    set(GLSLC_IS_GLSLANG_VALIDATOR TRUE)
endif()

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SPV_DIR    ${CMAKE_CURRENT_BINARY_DIR})

set(SHADERS
    relu.comp
    matmul.comp
    fc_layer.comp
)

set(SPV_OUTPUTS)
foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPV_OUT ${SPV_DIR}/${SHADER_NAME}.spv)

    if(GLSLC_IS_GLSLANG_VALIDATOR)
        add_custom_command(
            OUTPUT  ${SPV_OUT}
            COMMAND ${GLSLC} -V ${SHADER_DIR}/${SHADER} -o ${SPV_OUT}
            DEPENDS ${SHADER_DIR}/${SHADER}
            COMMENT "Compiling shader (glslangValidator): ${SHADER} -> ${SHADER_NAME}.spv"
        )
    else()
        add_custom_command(
            OUTPUT  ${SPV_OUT}
            COMMAND ${GLSLC} ${SHADER_DIR}/${SHADER} -o ${SPV_OUT}
            DEPENDS ${SHADER_DIR}/${SHADER}
            COMMENT "Compiling shader (glslc): ${SHADER} -> ${SHADER_NAME}.spv"
        )
    endif()

    list(APPEND SPV_OUTPUTS ${SPV_OUT})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_OUTPUTS})
