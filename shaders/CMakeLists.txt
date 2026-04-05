if(NOT Vulkan_FOUND OR NOT GLSLC)
    return()
endif()

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SPV_DIR    ${CMAKE_CURRENT_BINARY_DIR})

set(SHADERS
    relu.comp
    matmul.comp
)

set(SPV_OUTPUTS)
foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPV_OUT ${SPV_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT  ${SPV_OUT}
        COMMAND ${GLSLC} ${SHADER_DIR}/${SHADER} -o ${SPV_OUT}
        DEPENDS ${SHADER_DIR}/${SHADER}
        COMMENT "Compiling shader: ${SHADER} -> ${SHADER_NAME}.spv"
    )
    list(APPEND SPV_OUTPUTS ${SPV_OUT})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_OUTPUTS})
